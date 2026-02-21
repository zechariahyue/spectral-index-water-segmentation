"""
Step 5: Augmentation Ablation — Flips-Only (Reviewer 3 Point 5)
Addresses: "The augmentation pipeline effect is not quantified"

Identical to step3_train_fullscale_ablation.py EXCEPT:
  - Augmentation: horizontal flip + vertical flip ONLY (p=0.5 each)
  - Everything else unchanged (architecture, loss, optimizer, split)

Purpose: Show spectral index benefit persists under minimal augmentation.
Expected: F1 close to 99.17% (full pipeline), confirming indices are the driver.
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import numpy as np
import rasterio
from pathlib import Path
import json
from datetime import datetime
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass


class StandardSegmentationLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def dice_loss(self, pred, target):
        smooth = 1e-5
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        return 1 - ((2.0 * intersection + smooth) / (union + smooth)).mean()

    def focal_loss(self, pred, target):
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        p_t = torch.sigmoid(pred) * target + (1 - torch.sigmoid(pred)) * (1 - target)
        return (self.alpha * (1 - p_t) ** self.gamma * bce).mean()

    def forward(self, pred, target):
        return self.dice_loss(pred, target) + self.focal_loss(pred, target) + \
               F.binary_cross_entropy_with_logits(pred, target)


class S1S2WaterDataset(Dataset):
    def __init__(self, data_dir="../dataset/s1s2_water_minimal", mode='train',
                 sample_ids=None, tile_size=256, tiles_per_sample=None, augment=True):
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.tile_size = tile_size
        self.augment = augment and (mode == 'train')
        self.tiles = []

        if sample_ids is None:
            if mode == 'train':
                sample_ids = [i for i in range(1, 46) if i != 31]
            elif mode == 'val':
                sample_ids = list(range(46, 55))
            else:
                sample_ids = list(range(55, 66))

        if tiles_per_sample is None:
            tiles_per_sample = 200 if mode == 'train' else 100

        print(f"\n{'='*70}")
        print(f"S1S2-WATER DATASET ({mode.upper()}) - FLIPS-ONLY AUGMENTATION ABLATION")
        print(f"{'='*70}")
        print(f"Samples: {len(sample_ids)} | Tiles per sample: {tiles_per_sample} | Augment: {self.augment}")

        for sample_id in sample_ids:
            self._index_sample_tiles(sample_id, tiles_per_sample)

        print(f"Total tiles: {len(self.tiles)}\n")

    def _index_sample_tiles(self, sample_id, num_tiles):
        """Store file paths + coordinates only — no image data loaded here."""
        sample_dir = self.data_dir / str(sample_id)
        img_path = sample_dir / f"sentinel12_s2_{sample_id}_img.tif"
        mask_path = sample_dir / f"sentinel12_s1_{sample_id}_msk_aligned.tif"
        if not mask_path.exists():
            mask_path = sample_dir / f"sentinel12_s1_{sample_id}_msk.tif"
        if not img_path.exists() or not mask_path.exists():
            return
        try:
            with rasterio.open(img_path) as src:
                H, W = src.height, src.width
            if H <= self.tile_size or W <= self.tile_size:
                return
            for _ in range(num_tiles):
                y = np.random.randint(0, H - self.tile_size + 1)
                x = np.random.randint(0, W - self.tile_size + 1)
                self.tiles.append((str(img_path), str(mask_path), y, x))
        except Exception as e:
            print(f"[WARNING] Sample {sample_id}: {e}")

    def compute_spectral_indices(self, image):
        blue, green, red, nir, swir = image[0], image[1], image[2], image[3], image[4]
        ndwi  = ((green - nir)  / (green + nir  + 1e-8) + 1) / 2
        mndwi = ((green - swir) / (green + swir + 1e-8) + 1) / 2
        awei  = np.clip((4*(green - swir) - 0.25*nir + 2.75*swir + 2) / 4, 0, 1)
        return np.stack([blue, green, red, nir, swir, ndwi, mndwi, awei], axis=0)

    def apply_augmentation(self, image, mask):
        """FLIPS ONLY — horizontal flip (p=0.5) and vertical flip (p=0.5)."""
        if not self.augment:
            return image, mask
        if np.random.rand() < 0.5:
            image = np.flip(image, axis=2).copy()
            mask  = np.flip(mask,  axis=1).copy()
        if np.random.rand() < 0.5:
            image = np.flip(image, axis=1).copy()
            mask  = np.flip(mask,  axis=0).copy()
        return image, mask

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        img_path, mask_path, y, x = self.tiles[idx]
        window = rasterio.windows.Window(x, y, self.tile_size, self.tile_size)
        with rasterio.open(img_path) as src:
            tile_img = src.read(indexes=list(range(1, 6)), window=window).astype(np.float32)
        with rasterio.open(mask_path) as src:
            tile_mask = src.read(1, window=window).astype(np.float32)
        tile_img = np.clip(tile_img / 10000.0, 0, 1)
        tile_mask = (tile_mask > 0).astype(np.float32)
        tile_img, tile_mask = self.apply_augmentation(tile_img, tile_mask)
        image_8ch = self.compute_spectral_indices(tile_img)
        return torch.from_numpy(image_8ch).float(), torch.from_numpy(tile_mask).float().unsqueeze(0)


def create_model():
    model = smp.DeepLabV3Plus(encoder_name='resnet50', encoder_weights='imagenet',
                               in_channels=3, classes=1, activation=None)
    old_conv = model.encoder.conv1
    new_conv = nn.Conv2d(8, 64, kernel_size=7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        new_conv.weight[:, :3] = old_conv.weight
        nn.init.kaiming_normal_(new_conv.weight[:, 3:])
    model.encoder.conv1 = new_conv
    return model


def evaluate(model, loader, device):
    model.eval()
    preds, masks = [], []
    with torch.no_grad():
        for images, m in loader:
            out = torch.sigmoid(model(images.to(device)))
            preds.append(out.cpu().numpy())
            masks.append(m.numpy())
    preds = (np.concatenate(preds).flatten() > 0.5).astype(np.float32)
    masks = np.concatenate(masks).flatten()
    return {
        'f1':        f1_score(masks, preds, zero_division=0),
        'iou':       jaccard_score(masks, preds, zero_division=0),
        'precision': precision_score(masks, preds, zero_division=0),
        'recall':    recall_score(masks, preds, zero_division=0),
        'accuracy':  accuracy_score(masks, preds),
    }


def main():
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type == 'cuda':
            torch.zeros(1).to(device)  # test GPU is actually usable
    except Exception:
        device = torch.device('cpu')
    print(f"Device: {device}")
    print("\n" + "="*70)
    print("AUGMENTATION ABLATION: FLIPS-ONLY (Reviewer 3 Point 5)")
    print("="*70)
    print("Augmentation: horizontal flip + vertical flip ONLY")
    print("All other settings identical to step3 (full pipeline, 99.17% F1)")
    print("="*70 + "\n")

    train_ds = S1S2WaterDataset(mode='train', augment=True)
    val_ds   = S1S2WaterDataset(mode='val',   augment=False)
    test_ds  = S1S2WaterDataset(mode='test',  augment=False)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=8, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=8, shuffle=False, num_workers=0)

    model     = create_model().to(device)
    criterion = StandardSegmentationLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    best_val_f1, patience_counter, best_state = 0, 0, None
    max_epochs, patience = 20, 5

    for epoch in range(max_epochs):
        model.train()
        train_loss, n = 0, 0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item(); n += 1
        scheduler.step()

        val_metrics = evaluate(model, val_loader, device)
        val_f1 = val_metrics['f1']
        print(f"Epoch {epoch+1:2d}/{max_epochs} | Loss: {train_loss/n:.4f} | "
              f"Val F1: {val_f1*100:.2f}% | Val IoU: {val_metrics['iou']*100:.2f}%")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"  -> New best: {val_f1*100:.2f}%")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device)

    print("\n" + "="*70)
    print("TEST RESULTS (Flips-Only Augmentation):")
    print(f"  F1:        {test_metrics['f1']*100:.2f}%")
    print(f"  IoU:       {test_metrics['iou']*100:.2f}%")
    print(f"  Precision: {test_metrics['precision']*100:.2f}%")
    print(f"  Recall:    {test_metrics['recall']*100:.2f}%")
    print()
    print("COMPARISON:")
    print(f"  5ch baseline (full aug):    83.91%")
    print(f"  8ch flips-only (this run):  {test_metrics['f1']*100:.2f}%")
    print(f"  8ch full pipeline (step3):  99.17%")
    print("="*70)

    results = {
        'experiment': 'Augmentation Ablation — Flips-Only (Reviewer 3 Point 5)',
        'augmentation': 'horizontal_flip_p0.5 + vertical_flip_p0.5 only',
        'model': '8-channel DeepLabV3+ (ResNet-50)',
        'dataset': 'S1S2-Water (45 train / 9 val / 11 test)',
        'best_val_f1': best_val_f1,
        'test_metrics': test_metrics,
        'comparison': {
            '5ch_baseline_full_aug': 0.8391,
            '8ch_flips_only': test_metrics['f1'],
            '8ch_full_pipeline': 0.9917,
        },
        'timestamp': datetime.now().isoformat()
    }

    out_path = '../manuscript/augmentation_ablation_flipsonly_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == '__main__':
    main()

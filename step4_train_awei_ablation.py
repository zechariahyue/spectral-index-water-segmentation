"""
Step 4: AWEI Ablation Experiment (Reviewer 3 Point 3)
Addresses: "AWEI calibration or ablation needed to validate its contribution"

This script trains a 7-channel DeepLabV3+ model WITHOUT AWEI:
✅ 7 channels: RGB + NIR + SWIR + NDWI + MNDWI (NO AWEI)
❌ NO Spectral Consistency Loss (λ = 0.0)

Purpose: Determine if AWEI adds value beyond NDWI and MNDWI

Expected Outcome:
- 8ch model (with AWEI): 99.17% F1
- 7ch model (no AWEI): ???% F1
- Difference reveals AWEI's contribution

Training on full 45 samples with identical hyperparameters.
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

# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass


class StandardSegmentationLoss(nn.Module):
    """Standard loss WITHOUT spectral consistency"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def dice_loss(self, pred, target):
        smooth = 1e-5
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return 1 - dice.mean()

    def focal_loss(self, pred, target):
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pred_prob = torch.sigmoid(pred)
        p_t = pred_prob * target + (1 - pred_prob) * (1 - target)
        focal = self.alpha * (1 - p_t) ** self.gamma * bce
        return focal.mean()

    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred, target)
        bce = F.binary_cross_entropy_with_logits(pred, target)
        return dice + focal + bce


class S1S2WaterDataset(Dataset):
    """S1S2-Water dataset with 7 channels (NO AWEI)"""

    def __init__(self, data_dir="dataset/s1s2_water_minimal", mode='train',
                 sample_ids=None, tile_size=256, tiles_per_sample=None, augment=True):
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.tile_size = tile_size
        self.augment = augment and (mode == 'train')
        self.tiles = []

        if sample_ids is None:
            if mode == 'train':
                sample_ids = list(range(1, 46))
                if 31 in sample_ids:
                    sample_ids.remove(31)
            elif mode == 'val':
                sample_ids = list(range(46, 55))
            else:
                sample_ids = list(range(55, 66))

        self.sample_ids = sample_ids

        if tiles_per_sample is None:
            tiles_per_sample = 150 if mode == 'train' else 80

        print(f"\n{'='*70}")
        print(f"S1S2-WATER DATASET ({mode.upper()}) - AWEI ABLATION (7ch)")
        print(f"{'='*70}")
        print(f"Sample IDs: {sample_ids[:5]}... ({len(sample_ids)} total)")
        print(f"Target tiles per sample: {tiles_per_sample}")
        print(f"Augmentation: {self.augment}")

        for sample_id in sample_ids:
            self._load_sample_tiles(sample_id, tiles_per_sample)

        print(f"Total tiles loaded: {len(self.tiles)}")
        if len(self.tiles) > 0:
            water_ratios = [t['water_ratio'] for t in self.tiles]
            print(f"Water coverage: {np.mean(water_ratios)*100:.1f}% ± {np.std(water_ratios)*100:.1f}%")
        print(f"{'='*70}\n")

    def _load_sample_tiles(self, sample_id, num_tiles):
        sample_dir = self.data_dir / str(sample_id)
        s2_img_path = sample_dir / f"sentinel12_s2_{sample_id}_img.tif"
        s2_mask_path_aligned = sample_dir / f"sentinel12_s1_{sample_id}_msk_aligned.tif"
        s2_mask_path = sample_dir / f"sentinel12_s1_{sample_id}_msk.tif"

        if not s2_img_path.exists():
            return

        mask_path = s2_mask_path_aligned if s2_mask_path_aligned.exists() else s2_mask_path
        if not mask_path.exists():
            return

        try:
            with rasterio.open(s2_img_path) as src:
                s2_img = src.read()

            with rasterio.open(mask_path) as src:
                mask = src.read(1)

            rgb_nir_swir = s2_img[:5, :, :].astype(np.float32)
            rgb_nir_swir = np.clip(rgb_nir_swir / 10000.0, 0, 1)
            mask = (mask > 0).astype(np.float32)

            H, W = mask.shape

            for _ in range(num_tiles):
                if H <= self.tile_size or W <= self.tile_size:
                    continue

                y = np.random.randint(0, H - self.tile_size + 1)
                x = np.random.randint(0, W - self.tile_size + 1)

                tile_img = rgb_nir_swir[:, y:y+self.tile_size, x:x+self.tile_size]
                tile_mask = mask[y:y+self.tile_size, x:x+self.tile_size]

                if tile_img.shape[1] != self.tile_size or tile_img.shape[2] != self.tile_size:
                    continue

                self.tiles.append({
                    'image': tile_img,
                    'mask': tile_mask,
                    'water_ratio': np.mean(tile_mask),
                    'sample_id': sample_id
                })

        except Exception as e:
            print(f"[WARNING] Error loading sample {sample_id}: {e}")

    def compute_spectral_indices(self, image):
        """
        Compute NDWI and MNDWI only (NO AWEI)
        Input: (5, H, W) - [Blue, Green, Red, NIR, SWIR]
        Output: (7, H, W) - [Blue, Green, Red, NIR, SWIR, NDWI, MNDWI]
        """
        blue = image[0]
        green = image[1]
        red = image[2]
        nir = image[3]
        swir = image[4]

        # NDWI = (Green - NIR) / (Green + NIR)
        ndwi = (green - nir) / (green + nir + 1e-8)
        ndwi = (ndwi + 1) / 2

        # MNDWI = (Green - SWIR) / (Green + SWIR)
        mndwi = (green - swir) / (green + swir + 1e-8)
        mndwi = (mndwi + 1) / 2

        # Stack 7 channels (NO AWEI)
        image_7ch = np.stack([blue, green, red, nir, swir, ndwi, mndwi], axis=0)
        return image_7ch

    def apply_augmentation(self, image, mask):
        if not self.augment:
            return image, mask

        if np.random.rand() < 0.5:
            image = np.flip(image, axis=2).copy()
            mask = np.flip(mask, axis=1).copy()

        if np.random.rand() < 0.5:
            image = np.flip(image, axis=1).copy()
            mask = np.flip(mask, axis=0).copy()

        if np.random.rand() < 0.5:
            k = np.random.randint(1, 4)
            image = np.rot90(image, k, axes=(1, 2)).copy()
            mask = np.rot90(mask, k, axes=(0, 1)).copy()

        if np.random.rand() < 0.5:
            factor = np.random.uniform(0.7, 1.3)
            image[:5] = np.clip(image[:5] * factor, 0, 1)

        if np.random.rand() < 0.5:
            factor = np.random.uniform(0.7, 1.3)
            for i in range(5):
                mean = np.mean(image[i])
                image[i] = np.clip((image[i] - mean) * factor + mean, 0, 1)

        if np.random.rand() < 0.3:
            noise = np.random.normal(0, 0.02, image[:5].shape).astype(np.float32)
            image[:5] = np.clip(image[:5] + noise, 0, 1)

        if np.random.rand() < 0.4:
            gamma = np.random.uniform(0.8, 1.2)
            image[:5] = np.power(image[:5], gamma)

        return image, mask

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        tile = self.tiles[idx]
        image = tile['image'].copy()
        mask = tile['mask'].copy()

        image, mask = self.apply_augmentation(image, mask)
        image_7ch = self.compute_spectral_indices(image)

        image_tensor = torch.from_numpy(image_7ch).float()
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)

        return image_tensor, mask_tensor


def create_model(num_classes=1, in_channels=7):
    """Create DeepLabV3+ model with 7-channel input"""
    model = smp.DeepLabV3Plus(
        encoder_name='resnet50',
        encoder_weights='imagenet',
        in_channels=3,
        classes=num_classes,
        activation=None
    )

    old_conv = model.encoder.conv1
    new_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

    with torch.no_grad():
        new_conv.weight[:, :3, :, :] = old_conv.weight
        nn.init.kaiming_normal_(new_conv.weight[:, 3:, :, :])

    model.encoder.conv1 = new_conv
    return model


def train_awei_ablation():
    """Train 7-channel model WITHOUT AWEI"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("\n" + "="*80)
    print("AWEI ABLATION EXPERIMENT (Reviewer 3 Point 3)")
    print("="*80)
    print("Configuration:")
    print("  [OK] 7-channel input (RGB + NIR + SWIR + NDWI + MNDWI)")
    print("  [OK] NO AWEI channel")
    print("  [OK] NO Spectral Consistency Loss")
    print("  [OK] FULL 45 training samples")
    print()
    print("Purpose:")
    print("  Determine if AWEI adds value beyond NDWI and MNDWI")
    print()
    print("Expected Outcome:")
    print("  8ch model (with AWEI): 99.17% F1")
    print("  7ch model (no AWEI): ???% F1")
    print("  Difference reveals AWEI contribution")
    print("="*80 + "\n")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB\n")

    train_dataset = S1S2WaterDataset(mode='train', augment=True)
    val_dataset = S1S2WaterDataset(mode='val', augment=False)
    test_dataset = S1S2WaterDataset(mode='test', augment=False)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,
                             num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False,
                           num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False,
                            num_workers=0, pin_memory=False)

    model = create_model(in_channels=7).to(device)
    criterion = StandardSegmentationLoss(alpha=0.25, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    max_epochs = 20
    patience = 5
    best_val_f1 = 0
    patience_counter = 0

    print(f"Starting training...")
    print(f"Max epochs: {max_epochs}")
    print(f"Early stopping patience: {patience}")
    print(f"Batch size: 16\n")

    history = {'train_loss': [], 'val_f1': [], 'val_iou': []}

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0
        n_batches = 0

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1

        train_loss /= n_batches
        scheduler.step()

        # Validation with memory-efficient metric calculation
        model.eval()
        val_preds_list = []
        val_masks_list = []

        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = torch.sigmoid(model(images))
                val_preds_list.append((outputs.cpu().numpy() > 0.5).astype(np.float32))
                val_masks_list.append(masks.cpu().numpy())

        # Concatenate and flatten in chunks to avoid memory issues
        val_preds = np.concatenate(val_preds_list, axis=0).flatten()
        val_masks = np.concatenate(val_masks_list, axis=0).flatten()

        val_f1 = f1_score(val_masks, val_preds, zero_division=0)
        val_iou = jaccard_score(val_masks, val_preds, zero_division=0)
        val_precision = precision_score(val_masks, val_preds, zero_division=0)
        val_recall = recall_score(val_masks, val_preds, zero_division=0)

        history['train_loss'].append(train_loss)
        history['val_f1'].append(val_f1)
        history['val_iou'].append(val_iou)

        print(f"Epoch {epoch+1}/{max_epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val F1: {val_f1:.4f} ({val_f1*100:.2f}%) | Val IoU: {val_iou:.4f} ({val_iou*100:.2f}%)")
        print(f"  Val Precision: {val_precision:.4f} | Val Recall: {val_recall:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), 'manuscript/best_awei_ablation_7ch.pth')
            print(f"  [OK] New best model saved! F1: {val_f1*100:.2f}%")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"\n[OK] Early stopping triggered after {epoch+1} epochs")
            break
        print()

    # Test evaluation
    print("\n" + "="*80)
    print("FINAL EVALUATION ON TEST SET")
    print("="*80 + "\n")

    model.load_state_dict(torch.load('manuscript/best_awei_ablation_7ch.pth'))
    model.eval()

    test_preds_list = []
    test_masks_list = []

    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = torch.sigmoid(model(images))
            test_preds_list.append((outputs.cpu().numpy() > 0.5).astype(np.float32))
            test_masks_list.append(masks.cpu().numpy())

    test_preds = np.concatenate(test_preds_list, axis=0).flatten()
    test_masks = np.concatenate(test_masks_list, axis=0).flatten()

    test_f1 = f1_score(test_masks, test_preds, zero_division=0)
    test_iou = jaccard_score(test_masks, test_preds, zero_division=0)
    test_precision = precision_score(test_masks, test_preds, zero_division=0)
    test_recall = recall_score(test_masks, test_preds, zero_division=0)
    test_accuracy = accuracy_score(test_masks, test_preds)

    print("TEST SET RESULTS:")
    print(f"  F1-Score:  {test_f1:.4f} ({test_f1*100:.2f}%)")
    print(f"  IoU:       {test_iou:.4f} ({test_iou*100:.2f}%)")
    print(f"  Precision: {test_precision:.4f} ({test_precision*100:.2f}%)")
    print(f"  Recall:    {test_recall:.4f} ({test_recall*100:.2f}%)")
    print(f"  Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print()

    # Calculate AWEI contribution
    model_8ch_f1 = 0.9917  # From fullscale ablation
    awei_contribution = model_8ch_f1 - test_f1

    print("="*80)
    print("AWEI ABLATION ANALYSIS:")
    print("="*80)
    print(f"8ch model (with AWEI):    {model_8ch_f1*100:.2f}% F1")
    print(f"7ch model (no AWEI):      {test_f1*100:.2f}% F1")
    print()
    print(f"AWEI contribution:        {awei_contribution:+.4f} ({awei_contribution*100:+.2f}pp)")
    print()

    if abs(awei_contribution) < 0.001:
        interpretation = "AWEI provides NO significant benefit (<0.1pp)"
    elif awei_contribution > 0.005:
        interpretation = "AWEI provides SIGNIFICANT benefit (>0.5pp)"
    elif awei_contribution > 0:
        interpretation = "AWEI provides SMALL benefit (0.1-0.5pp)"
    else:
        interpretation = "AWEI HURTS performance (negative contribution)"

    print(f"INTERPRETATION: {interpretation}")
    print("="*80 + "\n")

    results = {
        'experiment': 'AWEI Ablation - 7ch model WITHOUT AWEI',
        'model': '7-channel DeepLabV3+ (ResNet-50, ImageNet init)',
        'input_channels': 'RGB + NIR + SWIR + NDWI + MNDWI (NO AWEI)',
        'loss_components': 'Dice + Focal + BCE (NO spectral consistency)',
        'dataset': 'S1S2-Water (45 train, 9 val, 11 test samples)',
        'best_val_f1': best_val_f1,
        'test_metrics': {
            'f1': test_f1,
            'iou': test_iou,
            'precision': test_precision,
            'recall': test_recall,
            'accuracy': test_accuracy
        },
        'awei_analysis': {
            '8ch_with_awei_f1': model_8ch_f1,
            '7ch_without_awei_f1': test_f1,
            'awei_contribution': awei_contribution,
            'interpretation': interpretation
        },
        'training_history': history,
        'timestamp': datetime.now().isoformat()
    }

    with open('manuscript/awei_ablation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("Results saved to: manuscript/awei_ablation_results.json")
    print("Model saved to: manuscript/best_awei_ablation_7ch.pth")
    print("="*80 + "\n")

    return results


if __name__ == '__main__':
    print("\n" + "="*80)
    print("AWEI ABLATION EXPERIMENT")
    print("="*80)
    print("This experiment will answer the question:")
    print("  'Does AWEI add value beyond NDWI and MNDWI?'")
    print()
    print("Expected runtime: ~2-3 hours")
    print("="*80 + "\n")

    print("Starting training automatically...")
    print()

    results = train_awei_ablation()

    print("\n" + "="*80)
    print("[OK] AWEI ABLATION EXPERIMENT COMPLETE")
    print("="*80)
    print("Next steps:")
    print("  1. Review results in: manuscript/awei_ablation_results.json")
    print("  2. Update Table 2 in manuscript with AWEI ablation results")
    print("  3. Address Reviewer 3 Point 3 in response letter")
    print("="*80 + "\n")

"""
AIWR Cross-Dataset Experiment (Reviewer 1, Point 8)
Trains 3-channel RGB baseline and 6-channel spectral-guided model on AIWR aerial dataset.
Compares against classical Green-Red proxy NDWI threshold.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score
from PIL import Image, ImageDraw
import warnings
warnings.filterwarnings('ignore')

DATASET_DIR = Path("dataset/AIWR Dataset")
RESULTS_FILE = Path("manuscript/aiwr_experiment_results.json")
IMG_SIZE = 512
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AIWRDataset(Dataset):
    def __init__(self, split='train', n_channels=3, augment=False):
        self.n_channels = n_channels
        self.augment = augment

        if split == 'test':
            img_dir = DATASET_DIR / 'Test'
            self.images = sorted(img_dir.glob('*.jpg'))
        else:
            img_dir = DATASET_DIR / 'Train'
            all_imgs = sorted(img_dir.glob('*.jpg'))
            np.random.seed(42)
            idx = np.random.permutation(len(all_imgs))
            n_val = max(1, int(0.1 * len(all_imgs)))
            if split == 'val':
                self.images = [all_imgs[i] for i in idx[-n_val:]]
            else:
                self.images = [all_imgs[i] for i in idx[:-n_val]]

        print(f"AIWR {split}: {len(self.images)} images, {n_channels}ch")

    def _load_mask(self, img_path):
        json_path = img_path.with_suffix('.json')
        if not json_path.exists():
            return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
        with open(json_path) as f:
            data = json.load(f)
        h, w = data.get('imageHeight', 650), data.get('imageWidth', 650)
        canvas = Image.new('L', (w, h), 0)
        draw = ImageDraw.Draw(canvas)
        for shape in data.get('shapes', []):
            if shape['shape_type'] == 'polygon' and len(shape['points']) >= 3:
                draw.polygon([(p[0], p[1]) for p in shape['points']], fill=255)
        mask = canvas.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
        return np.array(mask).astype(np.float32) / 255.0

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path).convert('RGB').resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        img = np.array(img).astype(np.float32) / 255.0
        mask = self._load_mask(img_path)

        img = img.transpose(2, 0, 1)  # (3, H, W)

        if self.augment:
            if np.random.rand() > 0.5:
                img = img[:, :, ::-1].copy()
                mask = mask[:, ::-1].copy()
            if np.random.rand() > 0.5:
                img = img[:, ::-1, :].copy()
                mask = mask[::-1, :].copy()

        if self.n_channels == 6:
            b, g, r = img[0], img[1], img[2]
            ndwi = (g - r) / (g + r + 1e-8)
            ndwi = (ndwi + 1) / 2
            mndwi = (b + g - 2 * r) / (b + g + 2 * r + 1e-8)
            mndwi = (mndwi + 1) / 2
            awei = np.clip((4 * g - 3 * r - 2 * b + 2) / 4, 0, 1)
            img = np.stack([b, g, r, ndwi, mndwi, awei], axis=0)

        return torch.from_numpy(img).float(), torch.from_numpy(mask).float().unsqueeze(0)


def make_model(n_channels):
    model = smp.DeepLabV3Plus(
        encoder_name='resnet50', encoder_weights='imagenet',
        in_channels=3, classes=1, activation=None
    )
    if n_channels != 3:
        old = model.encoder.conv1
        new_conv = nn.Conv2d(n_channels, 64, 7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            new_conv.weight[:, :3] = old.weight
            nn.init.kaiming_normal_(new_conv.weight[:, 3:])
        model.encoder.conv1 = new_conv
    return model.to(DEVICE)


def evaluate(model, loader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            out = torch.sigmoid(model(x.to(DEVICE))).cpu().numpy()
            preds.append(out.flatten())
            targets.append(y.numpy().flatten())
    p = np.concatenate(preds) > 0.5
    t = np.concatenate(targets)
    return {
        'f1': float(f1_score(t, p, zero_division=0)),
        'iou': float(jaccard_score(t, p, zero_division=0)),
        'precision': float(precision_score(t, p, zero_division=0)),
        'recall': float(recall_score(t, p, zero_division=0)),
    }


def train_model(n_channels, label):
    print(f"\n{'='*60}\nTraining: {label} ({n_channels}ch)\n{'='*60}")

    train_ds = AIWRDataset('train', n_channels, augment=True)
    val_ds   = AIWRDataset('val',   n_channels, augment=False)
    test_ds  = AIWRDataset('test',  n_channels, augment=False)

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=4, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=4, shuffle=False, num_workers=0)

    model = make_model(n_channels)
    criterion = smp.losses.DiceLoss(mode='binary')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    best_val_f1, best_state, no_improve = 0.0, None, 0
    patience = 7

    for epoch in range(40):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_f1 = evaluate(model, val_loader)['f1']
        print(f"Epoch {epoch+1:2d} | loss={total_loss/len(train_loader):.4f} | val_f1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader)
    print(f"\nTest F1: {test_metrics['f1']*100:.2f}%  IoU: {test_metrics['iou']*100:.2f}%")
    return test_metrics, best_val_f1


def classical_proxy_ndwi(test_ds):
    """Green-Red proxy NDWI threshold (threshold=0, i.e. Green > Red)"""
    preds, targets = [], []
    for img, mask in test_ds:
        g, r = img[1].numpy(), img[2].numpy()
        ndwi = (g - r) / (g + r + 1e-8)
        preds.append((ndwi > 0).flatten())
        targets.append(mask.numpy().flatten())
    p = np.concatenate(preds)
    t = np.concatenate(targets)
    return {
        'f1': float(f1_score(t, p, zero_division=0)),
        'iou': float(jaccard_score(t, p, zero_division=0)),
        'precision': float(precision_score(t, p, zero_division=0)),
        'recall': float(recall_score(t, p, zero_division=0)),
    }


if __name__ == '__main__':
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Classical baseline
    print("\nComputing classical Green-Red proxy NDWI baseline...")
    test_ds_3ch = AIWRDataset('test', n_channels=3, augment=False)
    classical = classical_proxy_ndwi(test_ds_3ch)
    print(f"Classical proxy NDWI F1: {classical['f1']*100:.2f}%")

    # 3-channel RGB CNN baseline
    baseline_metrics, baseline_val_f1 = train_model(3, "RGB Baseline")

    # 6-channel spectral-guided
    spectral_metrics, spectral_val_f1 = train_model(6, "Spectral-Guided (RGB + proxy indices)")

    results = {
        'dataset': 'AIWR Aerial Dataset (650x650 RGB, resized to 512x512)',
        'train_samples': 648,
        'test_samples': 80,
        'classical_proxy_ndwi': classical,
        'rgb_baseline_3ch': {'test': baseline_metrics, 'best_val_f1': baseline_val_f1},
        'spectral_guided_6ch': {'test': spectral_metrics, 'best_val_f1': spectral_val_f1},
        'delta_f1_pp': round((spectral_metrics['f1'] - baseline_metrics['f1']) * 100, 2),
        'timestamp': datetime.now().isoformat()
    }

    RESULTS_FILE.parent.mkdir(exist_ok=True)
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Classical proxy NDWI:  {classical['f1']*100:.2f}% F1")
    print(f"RGB baseline (3ch):    {baseline_metrics['f1']*100:.2f}% F1")
    print(f"Spectral-guided (6ch): {spectral_metrics['f1']*100:.2f}% F1")
    print(f"Delta:                 {(spectral_metrics['f1'] - baseline_metrics['f1'])*100:+.2f}pp")
    print(f"\nResults saved to: {RESULTS_FILE}")

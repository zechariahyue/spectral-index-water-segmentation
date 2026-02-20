"""
Step 7: SAR+Optical Fusion (Reviewer 3 Point 1)
Addresses: "No multi-modal fusion experiments to validate foundation claim"

This script trains a 10-channel DeepLabV3+ model using BOTH SAR and Optical data:
✅ Early fusion: 10 channels (8 optical + 2 SAR)
✅ Optical: RGB + NIR + SWIR + NDWI + MNDWI + AWEI
✅ SAR: VV + VH polarizations

Purpose: Validate "foundation for fusion" claim

Expected Outcome:
- Optical-only (8ch): 99.17% F1
- SAR-only (2ch): ???% F1
- Fusion (10ch): ???% F1
- If fusion > optical: SAR adds value
- If fusion ≈ optical: SAR redundant
- If fusion < optical: SAR hurts (unlikely)
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
from scipy.ndimage import zoom
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
    """Standard loss: Dice + Focal + BCE"""
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


class SAROpticalFusionDataset(Dataset):
    """
    S1S2-Water dataset with SAR+Optical fusion (10 channels)
    Early fusion: concatenate SAR and optical at input level
    """

    def __init__(self, data_dir="dataset/s1s2_water_minimal", mode='train',
                 sample_ids=None, tile_size=256, tiles_per_sample=None, augment=True):
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.tile_size = tile_size
        self.augment = augment and (mode == 'train')
        self.tiles = []

        # Use the exact split from manuscript
        if sample_ids is None:
            if mode == 'train':
                sample_ids = list(range(1, 46))
                if 31 in sample_ids:
                    sample_ids.remove(31)  # Corrupted sample
            elif mode == 'val':
                sample_ids = list(range(46, 55))
            else:  # test
                sample_ids = list(range(55, 66))

        self.sample_ids = sample_ids

        # Calculate tiles per sample (REDUCED for memory efficiency)
        if tiles_per_sample is None:
            # Reduced from 200/100 to 50/25 to prevent RAM overflow
            # Each sample loads full images temporarily (~3-4GB), so fewer samples loaded at once
            tiles_per_sample = 50 if mode == 'train' else 25

        print(f"\n{'='*70}")
        print(f"SAR+OPTICAL FUSION DATASET ({mode.upper()})")
        print(f"{'='*70}")
        print(f"Sample IDs: {sample_ids[:5]}... ({len(sample_ids)} total)")
        print(f"Target tiles per sample: {tiles_per_sample}")
        print(f"Augmentation: {self.augment}")

        for sample_id in sample_ids:
            self._load_sample_tiles(sample_id, tiles_per_sample)
            # Force garbage collection after each sample to free memory
            import gc
            gc.collect()

        print(f"Total tiles loaded: {len(self.tiles)}")
        if len(self.tiles) > 0:
            water_ratios = [t['water_ratio'] for t in self.tiles]
            print(f"Water coverage: {np.mean(water_ratios)*100:.1f}% ± {np.std(water_ratios)*100:.1f}%")
        print(f"{'='*70}\n")

    def _load_sample_tiles(self, sample_id, num_tiles):
        sample_dir = self.data_dir / str(sample_id)

        s1_img_path = sample_dir / f"sentinel12_s1_{sample_id}_img.tif"
        s2_img_path = sample_dir / f"sentinel12_s2_{sample_id}_img.tif"
        s2_mask_path_aligned = sample_dir / f"sentinel12_s1_{sample_id}_msk_aligned.tif"
        s2_mask_path = sample_dir / f"sentinel12_s1_{sample_id}_msk.tif"

        if not s1_img_path.exists() or not s2_img_path.exists():
            return

        mask_path = s2_mask_path_aligned if s2_mask_path_aligned.exists() else s2_mask_path
        if not mask_path.exists():
            return

        try:
            # Load Sentinel-1 SAR image (2 bands: VV, VH)
            with rasterio.open(s1_img_path) as src:
                s1_img = src.read()  # Shape: (2, H, W)

            # Load Sentinel-2 optical image (5 bands: RGB, NIR, SWIR)
            with rasterio.open(s2_img_path) as src:
                s2_img = src.read()  # Shape: (bands, H, W)

            # Load mask
            with rasterio.open(mask_path) as src:
                mask = src.read(1)  # Shape: (H, W)

            # SAR preprocessing
            # Diagnostic showed: 79% negative + 20% zero = ~99% are no-data fill values.
            # Only positive pixels are valid backscatter. Compute percentiles from valid
            # pixels only, then set no-data pixels to neutral 0.5 in normalised space.
            s1_img = s1_img.astype(np.float32)
            s1_normalized = np.full_like(s1_img, 0.5)  # default: neutral

            for i in range(s1_img.shape[0]):
                band = s1_img[i]
                valid = band > 0
                if valid.sum() < 100:
                    continue
                p2  = np.percentile(band[valid], 2)
                p98 = np.percentile(band[valid], 98)
                normed = (band - p2) / (p98 - p2 + 1e-8)
                normed = np.clip(normed, 0, 1)
                normed[~valid] = 0.5  # no-data → neutral
                s1_normalized[i] = normed

            # Align SAR to optical resolution if needed
            if s1_normalized.shape[1] != mask.shape[0]:
                scale_factor = mask.shape[0] / s1_normalized.shape[1]
                s1_aligned = np.zeros((2, mask.shape[0], mask.shape[1]), dtype=np.float32)
                for i in range(2):
                    s1_aligned[i] = zoom(s1_normalized[i], scale_factor, order=1)
                s1_normalized = s1_aligned

            # Optical preprocessing: Extract first 5 bands and normalize
            rgb_nir_swir = s2_img[:5, :, :].astype(np.float32)
            rgb_nir_swir = rgb_nir_swir / 10000.0
            rgb_nir_swir = np.clip(rgb_nir_swir, 0, 1)

            # Binary mask
            mask = (mask > 0).astype(np.float32)

            H, W = mask.shape

            # Random tile extraction
            for _ in range(num_tiles):
                if H <= self.tile_size or W <= self.tile_size:
                    continue

                max_y = H - self.tile_size
                max_x = W - self.tile_size
                y = np.random.randint(0, max_y + 1)
                x = np.random.randint(0, max_x + 1)

                tile_optical = rgb_nir_swir[:, y:y+self.tile_size, x:x+self.tile_size]
                tile_sar = s1_normalized[:, y:y+self.tile_size, x:x+self.tile_size]
                tile_mask = mask[y:y+self.tile_size, x:x+self.tile_size]

                if tile_optical.shape[1] != self.tile_size or tile_optical.shape[2] != self.tile_size:
                    continue

                water_ratio = np.mean(tile_mask)
                self.tiles.append({
                    'optical': tile_optical,  # Shape: (5, 256, 256)
                    'sar': tile_sar,          # Shape: (2, 256, 256)
                    'mask': tile_mask,        # Shape: (256, 256)
                    'water_ratio': water_ratio,
                    'sample_id': sample_id
                })

        except Exception as e:
            print(f"[WARNING] Error loading sample {sample_id}: {e}")
            import gc
            gc.collect()  # Force garbage collection to free memory

    def compute_spectral_indices(self, image):
        """
        Compute NDWI, MNDWI, AWEI from 5-channel optical input
        Input: (5, H, W) - [Blue, Green, Red, NIR, SWIR]
        Output: (8, H, W) - [Blue, Green, Red, NIR, SWIR, NDWI, MNDWI, AWEI]
        """
        blue = image[0]
        green = image[1]
        red = image[2]
        nir = image[3]
        swir = image[4]

        # NDWI = (Green - NIR) / (Green + NIR)
        ndwi = (green - nir) / (green + nir + 1e-8)
        ndwi = (ndwi + 1) / 2  # Normalize to [0, 1]

        # MNDWI = (Green - SWIR) / (Green + SWIR)
        mndwi = (green - swir) / (green + swir + 1e-8)
        mndwi = (mndwi + 1) / 2  # Normalize to [0, 1]

        # AWEI = 4 * (Green - SWIR) - 0.25 * NIR + 2.75 * SWIR
        awei = 4 * (green - swir) - 0.25 * nir + 2.75 * swir
        awei = np.clip((awei + 2) / 4, 0, 1)  # Normalize to [0, 1]

        # Stack all 8 optical channels
        image_8ch = np.stack([blue, green, red, nir, swir, ndwi, mndwi, awei], axis=0)
        return image_8ch

    def apply_augmentation(self, optical, sar, mask):
        """Apply augmentation to both optical and SAR data"""
        if not self.augment:
            return optical, sar, mask

        # 1. Random horizontal flip (p=0.5)
        if np.random.rand() < 0.5:
            optical = np.flip(optical, axis=2).copy()
            sar = np.flip(sar, axis=2).copy()
            mask = np.flip(mask, axis=1).copy()

        # 2. Random vertical flip (p=0.5)
        if np.random.rand() < 0.5:
            optical = np.flip(optical, axis=1).copy()
            sar = np.flip(sar, axis=1).copy()
            mask = np.flip(mask, axis=0).copy()

        # 3. Random 90-degree rotation
        if np.random.rand() < 0.5:
            k = np.random.randint(1, 4)
            optical = np.rot90(optical, k, axes=(1, 2)).copy()
            sar = np.rot90(sar, k, axes=(1, 2)).copy()
            mask = np.rot90(mask, k, axes=(0, 1)).copy()

        # 4. Brightness adjustment (optical only, p=0.5)
        if np.random.rand() < 0.5:
            factor = np.random.uniform(0.7, 1.3)
            optical = np.clip(optical * factor, 0, 1)

        # 5. Contrast adjustment (optical only, p=0.5)
        if np.random.rand() < 0.5:
            factor = np.random.uniform(0.7, 1.3)
            for i in range(5):
                mean = np.mean(optical[i])
                optical[i] = np.clip((optical[i] - mean) * factor + mean, 0, 1)

        # 6. Gaussian noise (optical, p=0.3)
        if np.random.rand() < 0.3:
            noise = np.random.normal(0, 0.02, optical.shape)
            optical = np.clip(optical + noise, 0, 1)

        # 7. Speckle noise (SAR, p=0.3)
        if np.random.rand() < 0.3:
            noise = np.random.normal(0, 0.02, sar.shape)
            sar = np.clip(sar + noise, 0, 1)

        # 8. Gamma correction (optical only, p=0.4)
        if np.random.rand() < 0.4:
            gamma = np.random.uniform(0.8, 1.2)
            optical = np.power(optical, gamma)

        return optical, sar, mask

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        tile = self.tiles[idx]

        # Get optical and SAR images
        optical = tile['optical'].copy()  # (5, H, W)
        sar = tile['sar'].copy()          # (2, H, W)
        mask = tile['mask'].copy()        # (H, W)

        # Apply augmentation to raw data
        optical, sar, mask = self.apply_augmentation(optical, sar, mask)

        # Compute spectral indices AFTER augmentation
        optical_8ch = self.compute_spectral_indices(optical)

        # Concatenate optical (8ch) + SAR (2ch) = 10ch
        fusion_10ch = np.concatenate([optical_8ch, sar], axis=0)

        # Convert to tensors
        image_tensor = torch.from_numpy(fusion_10ch).float()
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)

        return image_tensor, mask_tensor


def create_model(num_classes=1, in_channels=10):
    """
    Create DeepLabV3+ model with 10-channel input (8 optical + 2 SAR)
    """
    model = smp.DeepLabV3Plus(
        encoder_name='resnet50',
        encoder_weights='imagenet',
        in_channels=3,  # Start with 3 for ImageNet weights
        classes=num_classes,
        activation=None
    )

    # Modify first conv layer for 10 channels
    old_conv = model.encoder.conv1
    new_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # Transfer ImageNet weights for first 3 channels (RGB)
    with torch.no_grad():
        new_conv.weight[:, :3, :, :] = old_conv.weight
        # Kaiming initialization for remaining 7 channels
        nn.init.kaiming_normal_(new_conv.weight[:, 3:, :, :])

    model.encoder.conv1 = new_conv
    return model


def train_sar_optical_fusion():
    """
    Train 10-channel SAR+Optical fusion model
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("\n" + "="*80)
    print("SAR+OPTICAL FUSION EXPERIMENT (Reviewer 3 Point 1)")
    print("="*80)
    print("Configuration:")
    print("  [OK] 10-channel input (8 optical + 2 SAR)")
    print("  [OK] Optical: RGB + NIR + SWIR + NDWI + MNDWI + AWEI")
    print("  [OK] SAR: VV + VH")
    print("  [OK] Early fusion architecture")
    print("  [OK] Standard loss (Dice + Focal + BCE)")
    print("  [OK] FULL 45 training samples")
    print()
    print("Purpose:")
    print("  Validate 'foundation for fusion' claim")
    print()
    print("Expected Outcome:")
    print("  Optical-only (8ch): 99.17% F1")
    print("  SAR-only (2ch): ???% F1")
    print("  Fusion (10ch): ???% F1")
    print("="*80 + "\n")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB\n")
        torch.cuda.empty_cache()  # Clear any leftover GPU state from previous runs

    # Load datasets
    train_dataset = SAROpticalFusionDataset(mode='train', augment=True)
    val_dataset = SAROpticalFusionDataset(mode='val', augment=False)
    test_dataset = SAROpticalFusionDataset(mode='test', augment=False)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True,
                             num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False,
                           num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False,
                            num_workers=0, pin_memory=False)

    # Model
    model = create_model(in_channels=10).to(device)

    # Loss function
    criterion = StandardSegmentationLoss(alpha=0.25, gamma=2.0)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    # Training parameters
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
        # Training
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

        # Validation
        model.eval()
        val_preds = []
        val_masks = []

        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = torch.sigmoid(model(images))
                val_preds.append(outputs.cpu().numpy())
                val_masks.append(masks.cpu().numpy())

        val_preds = np.concatenate(val_preds, axis=0).flatten()
        val_masks = np.concatenate(val_masks, axis=0).flatten()
        val_preds_binary = (val_preds > 0.5).astype(np.float32)

        val_f1 = f1_score(val_masks, val_preds_binary, zero_division=0)
        val_iou = jaccard_score(val_masks, val_preds_binary, zero_division=0)
        val_precision = precision_score(val_masks, val_preds_binary, zero_division=0)
        val_recall = recall_score(val_masks, val_preds_binary, zero_division=0)

        history['train_loss'].append(train_loss)
        history['val_f1'].append(val_f1)
        history['val_iou'].append(val_iou)

        print(f"Epoch {epoch+1}/{max_epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val F1: {val_f1:.4f} ({val_f1*100:.2f}%) | Val IoU: {val_iou:.4f} ({val_iou*100:.2f}%)")
        print(f"  Val Precision: {val_precision:.4f} | Val Recall: {val_recall:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), 'manuscript/best_sar_optical_fusion_10ch.pth')
            print(f"  [OK] New best model saved! F1: {val_f1*100:.2f}%")
            patience_counter = 0
        else:
            patience_counter += 1

        # Save model at least once (even if F1 is 0)
        if epoch == 0:
            torch.save(model.state_dict(), 'manuscript/best_sar_optical_fusion_10ch.pth')

        if patience_counter >= patience:
            print(f"\n[OK] Early stopping triggered after {epoch+1} epochs")
            break
        print()

    # Test evaluation
    print("\n" + "="*80)
    print("FINAL EVALUATION ON TEST SET")
    print("="*80 + "\n")

    model.load_state_dict(torch.load('manuscript/best_sar_optical_fusion_10ch.pth'))
    model.eval()

    test_preds = []
    test_masks = []

    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = torch.sigmoid(model(images))
            test_preds.append(outputs.cpu().numpy())
            test_masks.append(masks.cpu().numpy())

    test_preds = np.concatenate(test_preds, axis=0).flatten()
    test_masks = np.concatenate(test_masks, axis=0).flatten()
    test_preds_binary = (test_preds > 0.5).astype(np.float32)

    test_f1 = f1_score(test_masks, test_preds_binary, zero_division=0)
    test_iou = jaccard_score(test_masks, test_preds_binary, zero_division=0)
    test_precision = precision_score(test_masks, test_preds_binary, zero_division=0)
    test_recall = recall_score(test_masks, test_preds_binary, zero_division=0)
    test_accuracy = accuracy_score(test_masks, test_preds_binary)

    print("TEST SET RESULTS:")
    print(f"  F1-Score:  {test_f1:.4f} ({test_f1*100:.2f}%)")
    print(f"  IoU:       {test_iou:.4f} ({test_iou*100:.2f}%)")
    print(f"  Precision: {test_precision:.4f} ({test_precision*100:.2f}%)")
    print(f"  Recall:    {test_recall:.4f} ({test_recall*100:.2f}%)")
    print(f"  Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print()

    # Comprehensive comparison
    optical_only_f1 = 0.9917

    print("="*80)
    print("COMPREHENSIVE FUSION ANALYSIS:")
    print("="*80)
    print(f"Optical-only (8ch): {optical_only_f1*100:.2f}% F1")
    print(f"Fusion (10ch):      {test_f1*100:.2f}% F1")
    print(f"Fusion gain:        {(test_f1 - optical_only_f1)*100:+.2f}pp")
    print()

    if test_f1 > optical_only_f1 + 0.005:
        interpretation = "SAR adds SIGNIFICANT value (+0.5pp or more)"
    elif test_f1 > optical_only_f1:
        interpretation = "SAR adds SMALL value (0-0.5pp)"
    elif abs(test_f1 - optical_only_f1) < 0.003:
        interpretation = "SAR is REDUNDANT (no significant difference)"
    else:
        interpretation = "SAR HURTS performance (negative contribution)"

    print(f"INTERPRETATION: {interpretation}")
    print("="*80 + "\n")

    # Save results
    results = {
        'experiment': 'SAR+Optical Fusion - Early Fusion',
        'model': '10-channel DeepLabV3+ (ResNet-50, ImageNet init)',
        'input_channels': 'RGB + NIR + SWIR + NDWI + MNDWI + AWEI + VV + VH',
        'fusion_strategy': 'Early fusion (concatenate at input)',
        'loss_components': 'Dice + Focal + BCE',
        'dataset': 'S1S2-Water (45 train, 9 val, 11 test samples)',
        'best_val_f1': best_val_f1,
        'test_metrics': {
            'f1': test_f1,
            'iou': test_iou,
            'precision': test_precision,
            'recall': test_recall,
            'accuracy': test_accuracy
        },
        'fusion_analysis': {
            'optical_only_f1': optical_only_f1,
            'fusion_f1': test_f1,
            'fusion_gain': test_f1 - optical_only_f1,
            'interpretation': interpretation
        },
        'training_history': history,
        'timestamp': datetime.now().isoformat()
    }

    with open('manuscript/sar_optical_fusion_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("Results saved to: manuscript/sar_optical_fusion_results.json")
    print("Model saved to: manuscript/best_sar_optical_fusion_10ch.pth")
    print("="*80 + "\n")

    return results


if __name__ == '__main__':
    print("\n" + "="*80)
    print("SAR+OPTICAL FUSION EXPERIMENT")
    print("="*80)
    print("This experiment validates the 'foundation for fusion' claim")
    print()
    print("Expected runtime: ~3-4 hours")
    print("="*80 + "\n")

    print("Starting training automatically...")
    print()

    results = train_sar_optical_fusion()

    print("\n" + "="*80)
    print("[OK] SAR+OPTICAL FUSION EXPERIMENT COMPLETE")
    print("="*80)
    print("Next steps:")
    print("  1. Review results in: manuscript/sar_optical_fusion_results.json")
    print("  2. Compare all three models: Optical, SAR, Fusion")
    print("  3. Update manuscript with fusion results")
    print("  4. Address Reviewer 3 Point 1 in response letter")
    print("="*80 + "\n")

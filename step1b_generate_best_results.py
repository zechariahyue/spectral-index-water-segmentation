"""
Step 1B: Generate Best Qualitative Results (Parallel Processing)
Processes all test tiles in parallel, then selects best examples for manuscript

This version:
1. Runs inference on ALL test tiles
2. Categorizes results by scenario type
3. Selects best examples for each category
4. Generates high-quality comparison figures
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import rasterio
from rasterio.windows import Window
import torch
import torch.nn.functional as F
from pathlib import Path
import json
from typing import Tuple, Dict, List
import segmentation_models_pytorch as smp
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

# Configuration
DATA_DIR = Path("dataset/s1s2_water_minimal")
MODEL_PATH = Path("best_model_spectral-guided.pth")
OUTPUT_DIR = Path("manuscript/figures/qualitative_results_best")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEST_SAMPLES = [55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65]
TILE_SIZE = 256
NUM_WORKERS = 4  # Parallel workers for data loading

# Selection criteria for best examples
EXAMPLES_PER_CATEGORY = 2  # Number of examples per scenario type


def load_model(model_path: Path, device: str = 'cuda') -> torch.nn.Module:
    """Load the trained 8-channel spectral-guided model."""
    model = smp.DeepLabV3Plus(
        encoder_name="resnet50",
        encoder_weights=None,
        in_channels=8,
        classes=1,
        activation=None
    )

    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    return model


def compute_spectral_indices(bands: np.ndarray) -> np.ndarray:
    """Compute NDWI, MNDWI, AWEI."""
    blue, green, red, nir, swir = bands
    eps = 1e-8

    ndwi = (green - nir) / (green + nir + eps)
    ndwi_norm = (ndwi + 1) / 2

    mndwi = (green - swir) / (green + swir + eps)
    mndwi_norm = (mndwi + 1) / 2

    awei = 4 * (green - swir) - 0.25 * nir + 2.75 * swir
    awei_norm = np.clip((awei + 2) / 4, 0, 1)

    return np.stack([ndwi_norm, mndwi_norm, awei_norm], axis=0)


def load_tile(sample_id: int, tile_row: int, tile_col: int, data_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a single tile."""
    sample_dir = data_dir / str(sample_id)
    s2_img_path = sample_dir / f"sentinel12_s2_{sample_id}_img.tif"
    s1_mask_path_aligned = sample_dir / f"sentinel12_s1_{sample_id}_msk_aligned.tif"
    s1_mask_path = s1_mask_path_aligned if s1_mask_path_aligned.exists() else \
                   sample_dir / f"sentinel12_s1_{sample_id}_msk.tif"

    window = Window(tile_col * TILE_SIZE, tile_row * TILE_SIZE, TILE_SIZE, TILE_SIZE)

    with rasterio.open(s2_img_path) as src:
        bands = src.read([1, 2, 3, 4, 5], window=window).astype(np.float32)
        bands = bands / 10000.0
        bands = np.clip(bands, 0, 1)

    with rasterio.open(s1_mask_path) as src:
        mask = src.read(1, window=window).astype(np.float32)

    green, nir = bands[1], bands[3]
    ndwi = (green - nir) / (green + nir + 1e-8)

    return bands, mask, ndwi


def predict_tile(model: torch.nn.Module, bands: np.ndarray, device: str = 'cuda') -> np.ndarray:
    """Run model inference."""
    indices = compute_spectral_indices(bands)
    input_tensor = np.concatenate([bands, indices], axis=0)
    input_tensor = torch.from_numpy(input_tensor).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.sigmoid(logits)
        prediction = (probs > 0.5).cpu().numpy()[0, 0]

    return prediction.astype(np.float32)


def compute_metrics(prediction: np.ndarray, ground_truth: np.ndarray) -> Dict[str, float]:
    """Compute F1, IoU, Precision, Recall."""
    tp = np.sum((prediction == 1) & (ground_truth == 1))
    fp = np.sum((prediction == 1) & (ground_truth == 0))
    fn = np.sum((prediction == 0) & (ground_truth == 1))

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)

    return {'f1': f1 * 100, 'iou': iou * 100, 'precision': precision * 100, 'recall': recall * 100}


def get_all_tiles(sample_id: int, data_dir: Path) -> List[Tuple[int, int]]:
    """Get all valid tiles for a sample."""
    sample_dir = data_dir / str(sample_id)
    s1_mask_path_aligned = sample_dir / f"sentinel12_s1_{sample_id}_msk_aligned.tif"
    s1_mask_path = s1_mask_path_aligned if s1_mask_path_aligned.exists() else \
                   sample_dir / f"sentinel12_s1_{sample_id}_msk.tif"

    with rasterio.open(s1_mask_path) as src:
        h, w = src.shape

    n_rows = h // TILE_SIZE
    n_cols = w // TILE_SIZE

    tiles = []
    for row in range(n_rows):
        for col in range(n_cols):
            tiles.append((row, col))

    return tiles


def process_single_tile(args):
    """Process a single tile (for parallel execution)."""
    sample_id, tile_row, tile_col, data_dir, model_path, device = args

    try:
        # Load data
        bands, mask, ndwi = load_tile(sample_id, tile_row, tile_col, data_dir)

        # Check if tile has enough water (10-90% for interesting boundaries)
        water_frac = np.mean(mask)
        if water_frac < 0.1 or water_frac > 0.9:
            return None  # Skip tiles with too little or too much water

        # NDWI prediction
        ndwi_pred = (ndwi > 0.15).astype(np.float32)

        # Model prediction (load model per worker to avoid serialization issues)
        model = load_model(model_path, device)
        model_pred = predict_tile(model, bands, device)

        # Compute metrics
        ndwi_metrics = compute_metrics(ndwi_pred, mask)
        model_metrics = compute_metrics(model_pred, mask)

        improvement = model_metrics['f1'] - ndwi_metrics['f1']

        # Categorize scenario
        if improvement > 10:
            category = "model_wins_big"
        elif improvement > 2:
            category = "model_wins_small"
        elif improvement > -2:
            category = "tie"
        elif improvement > -10:
            category = "ndwi_wins_small"
        else:
            category = "ndwi_wins_big"

        # Determine if both methods struggle (both < 80% F1)
        if ndwi_metrics['f1'] < 80 and model_metrics['f1'] < 80:
            category = "both_struggle"

        # Determine if both methods excel (both > 95% F1)
        if ndwi_metrics['f1'] > 95 and model_metrics['f1'] > 95:
            category = "both_excel"

        return {
            'sample_id': sample_id,
            'tile_row': tile_row,
            'tile_col': tile_col,
            'bands': bands,
            'mask': mask,
            'ndwi': ndwi,
            'ndwi_pred': ndwi_pred,
            'model_pred': model_pred,
            'ndwi_f1': ndwi_metrics['f1'],
            'model_f1': model_metrics['f1'],
            'improvement': improvement,
            'category': category,
            'water_frac': water_frac
        }

    except Exception as e:
        return None


def create_false_color(bands: np.ndarray) -> np.ndarray:
    """Create NIR-Red-Green false-color composite."""
    nir, red, green = bands[3], bands[2], bands[1]
    composite = np.stack([nir, red, green], axis=-1)
    p2, p98 = np.percentile(composite, (2, 98))
    composite = np.clip((composite - p2) / (p98 - p2 + 1e-8), 0, 1)
    return composite


def create_error_map(prediction: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
    """Create RGB error map."""
    tp = (prediction == 1) & (ground_truth == 1)
    tn = (prediction == 0) & (ground_truth == 0)
    fp = (prediction == 1) & (ground_truth == 0)
    fn = (prediction == 0) & (ground_truth == 1)

    error_map = np.zeros((*prediction.shape, 3), dtype=np.float32)
    error_map[tp] = [0.2, 0.8, 0.2]
    error_map[tn] = [0.9, 0.9, 0.9]
    error_map[fp] = [1.0, 0.3, 0.3]
    error_map[fn] = [1.0, 0.8, 0.0]

    return error_map


def visualize_comparison(result: Dict, save_path: Path, title: str = ""):
    """Create comprehensive comparison figure."""
    fig = plt.figure(figsize=(20, 8))
    gs = GridSpec(2, 5, figure=fig, hspace=0.3, wspace=0.3)

    bands = result['bands']
    ndwi = result['ndwi']
    ndwi_pred = result['ndwi_pred']
    model_pred = result['model_pred']
    ground_truth = result['mask']

    # Row 1
    ax1 = fig.add_subplot(gs[0, 0])
    false_color = create_false_color(bands)
    ax1.imshow(false_color)
    ax1.set_title("(a) Sentinel-2 False-Color\n(NIR-Red-Green)", fontsize=10, fontweight='bold')
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(ndwi, cmap='RdYlBu', vmin=-1, vmax=1)
    ax2.set_title("(b) NDWI Index", fontsize=10, fontweight='bold')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(ndwi_pred, cmap='Blues', vmin=0, vmax=1)
    ax3.set_title("(c) NDWI Prediction\n(threshold=0.15)", fontsize=10, fontweight='bold')
    ax3.axis('off')

    ax4 = fig.add_subplot(gs[0, 3])
    ax4.imshow(model_pred, cmap='Blues', vmin=0, vmax=1)
    ax4.set_title("(d) Model Prediction\n(Spectral-guided DL)", fontsize=10, fontweight='bold')
    ax4.axis('off')

    ax5 = fig.add_subplot(gs[0, 4])
    ax5.imshow(ground_truth, cmap='Blues', vmin=0, vmax=1)
    ax5.set_title("(e) Ground Truth", fontsize=10, fontweight='bold')
    ax5.axis('off')

    # Row 2
    ax6 = fig.add_subplot(gs[1, 1])
    ndwi_error = create_error_map(ndwi_pred, ground_truth)
    ax6.imshow(ndwi_error)
    ax6.set_title("(f) NDWI Error Map", fontsize=10, fontweight='bold')
    ax6.axis('off')

    ax7 = fig.add_subplot(gs[1, 2])
    model_error = create_error_map(model_pred, ground_truth)
    ax7.imshow(model_error)
    ax7.set_title("(g) Model Error Map", fontsize=10, fontweight='bold')
    ax7.axis('off')

    ax8 = fig.add_subplot(gs[1, 3])
    ndwi_correct = (ndwi_pred == ground_truth)
    model_correct = (model_pred == ground_truth)
    improvement = np.zeros((*model_pred.shape, 3), dtype=np.float32)
    improvement[model_correct & ~ndwi_correct] = [0.2, 0.8, 0.2]
    improvement[~model_correct & ndwi_correct] = [1.0, 0.3, 0.3]
    improvement[model_correct & ndwi_correct] = [0.9, 0.9, 0.9]
    ax8.imshow(improvement)
    ax8.set_title("(h) Improvement Map\n(Green=Model Better)", fontsize=10, fontweight='bold')
    ax8.axis('off')

    ax9 = fig.add_subplot(gs[1, 4])
    ax9.axis('off')
    legend_elements = [
        mpatches.Patch(color=[0.2, 0.8, 0.2], label='True Positive'),
        mpatches.Patch(color=[0.9, 0.9, 0.9], label='True Negative'),
        mpatches.Patch(color=[1.0, 0.3, 0.3], label='False Positive'),
        mpatches.Patch(color=[1.0, 0.8, 0.0], label='False Negative')
    ]
    ax9.legend(handles=legend_elements, loc='center', fontsize=10, frameon=False)

    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main execution."""
    print("=" * 80)
    print("STEP 1B: GENERATING BEST QUALITATIVE RESULTS (PARALLEL)")
    print("=" * 80)
    print()

    if not MODEL_PATH.exists():
        print(f"[ERROR] Model not found at {MODEL_PATH}")
        return

    if not DATA_DIR.exists():
        print(f"[ERROR] Dataset not found at {DATA_DIR}")
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Parallel workers: {NUM_WORKERS}")
    print()

    # Collect all tiles from test samples
    print("Collecting all test tiles...")
    all_tile_args = []
    for sample_id in TEST_SAMPLES:
        sample_dir = DATA_DIR / str(sample_id)
        if not sample_dir.exists():
            continue

        try:
            tiles = get_all_tiles(sample_id, DATA_DIR)
            for tile_row, tile_col in tiles:
                all_tile_args.append((sample_id, tile_row, tile_col, DATA_DIR, MODEL_PATH, device))
        except Exception as e:
            print(f"  [WARNING] Error collecting tiles from sample {sample_id}: {e}")
            continue

    print(f"Total tiles to process: {len(all_tile_args)}")
    print()

    # Process all tiles in parallel
    print("Processing all tiles in parallel...")
    results = []

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(process_single_tile, args) for args in all_tile_args]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            result = future.result()
            if result is not None:
                results.append(result)

    print(f"\nProcessed {len(results)} valid tiles")
    print()

    # Categorize results
    categories = {}
    for result in results:
        cat = result['category']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(result)

    # Print category statistics
    print("Category Statistics:")
    print("-" * 80)
    for cat, items in sorted(categories.items()):
        print(f"  {cat}: {len(items)} tiles")
    print()

    # Select best examples from each category
    print("Selecting best examples from each category...")
    selected_results = []

    for cat, items in categories.items():
        # Sort by absolute improvement (most dramatic examples)
        items_sorted = sorted(items, key=lambda x: abs(x['improvement']), reverse=True)

        # Take top N examples
        selected = items_sorted[:EXAMPLES_PER_CATEGORY]
        selected_results.extend(selected)

        print(f"  {cat}: Selected {len(selected)} examples")

    print()
    print(f"Total selected: {len(selected_results)} examples")
    print()

    # Generate figures for selected examples
    print("Generating figures...")
    print("-" * 80)

    for i, result in enumerate(tqdm(selected_results, desc="Generating")):
        category = result['category']
        sample_id = result['sample_id']
        tile_row = result['tile_row']
        tile_col = result['tile_col']

        save_path = OUTPUT_DIR / f"{category}_sample{sample_id}_tile{tile_row}_{tile_col}.png"

        title = (f"{category.replace('_', ' ').title()} - Sample {sample_id} Tile ({tile_row}, {tile_col})\n"
                f"NDWI F1: {result['ndwi_f1']:.2f}% | "
                f"Model F1: {result['model_f1']:.2f}% | "
                f"Improvement: {result['improvement']:+.2f}pp")

        visualize_comparison(result, save_path, title)

    # Save summary
    summary = {
        'total_tiles_processed': len(results),
        'total_figures_generated': len(selected_results),
        'categories': {cat: len(items) for cat, items in categories.items()},
        'selected_examples': [
            {
                'category': r['category'],
                'sample_id': r['sample_id'],
                'tile_row': r['tile_row'],
                'tile_col': r['tile_col'],
                'ndwi_f1': r['ndwi_f1'],
                'model_f1': r['model_f1'],
                'improvement': r['improvement']
            }
            for r in selected_results
        ]
    }

    summary_path = OUTPUT_DIR / "best_results_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"Total tiles processed: {len(results)}")
    print(f"Total figures generated: {len(selected_results)}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    print("[OK] Best qualitative results generation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

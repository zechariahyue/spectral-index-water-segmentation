"""
Step 2: Quantify Dataset Diversity Metrics
Addresses Reviewer 3 Point 2: Provide measurable criteria for subset characteristics

Computes quantitative metrics comparing 8-sample ablation subset vs full 45-sample training set.
"""

import sys
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import rasterio
from scipy import stats
import json
from tqdm import tqdm

# Configuration
DATA_DIR = Path("dataset/s1s2_water_minimal")
OUTPUT_DIR = Path("manuscript/diversity_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Sample splits (based on your training scripts)
TRAIN_8_SAMPLES = [1, 5, 10, 15, 20, 25, 30, 35]  # 8-sample ablation subset
TRAIN_45_SAMPLES = list(range(1, 46))  # Full training set
TEST_SAMPLES = list(range(55, 66))  # Test set for reference


def compute_water_fraction(mask: np.ndarray) -> float:
    """Compute percentage of water pixels."""
    return float(np.mean(mask) * 100)


def compute_spectral_statistics(bands: np.ndarray) -> dict:
    """
    Compute spectral statistics.

    Args:
        bands: (5, H, W) array with [Blue, Green, Red, NIR, SWIR]
    """
    blue, green, red, nir, swir = bands
    eps = 1e-8

    # Compute NDWI
    ndwi = (green - nir) / (green + nir + eps)

    # Compute MNDWI
    mndwi = (green - swir) / (green + swir + eps)

    return {
        'ndwi_mean': float(np.mean(ndwi)),
        'ndwi_std': float(np.std(ndwi)),
        'ndwi_min': float(np.min(ndwi)),
        'ndwi_max': float(np.max(ndwi)),
        'mndwi_mean': float(np.mean(mndwi)),
        'mndwi_std': float(np.std(mndwi)),
        'nir_mean': float(np.mean(nir)),
        'nir_std': float(np.std(nir)),
        'swir_mean': float(np.mean(swir)),
        'swir_std': float(np.std(swir))
    }


def compute_edge_density(mask: np.ndarray) -> float:
    """Compute edge density as measure of scene complexity."""
    from scipy.ndimage import sobel

    grad_x = sobel(mask.astype(float), axis=0)
    grad_y = sobel(mask.astype(float), axis=1)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    edge_pixels = gradient_magnitude > 0.1
    edge_density = float(np.mean(edge_pixels) * 100)

    return edge_density


def compute_ndwi_baseline_performance(bands: np.ndarray, mask: np.ndarray, threshold: float = 0.15) -> dict:
    """Compute NDWI baseline F1-score (difficulty metric)."""
    green, nir = bands[1], bands[3]
    ndwi = (green - nir) / (green + nir + 1e-8)
    ndwi_pred = (ndwi > threshold).astype(int)

    y_true = mask.flatten()
    y_pred = ndwi_pred.flatten()

    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        'ndwi_f1': float(f1 * 100),
        'ndwi_precision': float(precision * 100),
        'ndwi_recall': float(recall * 100)
    }


def analyze_sample(sample_id: int, data_dir: Path) -> dict:
    """Comprehensive analysis of a single sample."""
    sample_dir = data_dir / str(sample_id)
    s2_img_path = sample_dir / f"sentinel12_s2_{sample_id}_img.tif"
    s1_mask_path_aligned = sample_dir / f"sentinel12_s1_{sample_id}_msk_aligned.tif"
    s1_mask_path = s1_mask_path_aligned if s1_mask_path_aligned.exists() else \
                   sample_dir / f"sentinel12_s1_{sample_id}_msk.tif"

    if not (s2_img_path.exists() and s1_mask_path.exists()):
        return None

    try:
        # Load data (downsample for speed)
        with rasterio.open(s2_img_path) as src:
            # Read at 1/4 resolution for faster processing
            bands = src.read(
                [1, 2, 3, 4, 5],
                out_shape=(5, src.height // 4, src.width // 4)
            ).astype(np.float32) / 10000.0
            bands = np.clip(bands, 0, 1)

        with rasterio.open(s1_mask_path) as src:
            mask = src.read(
                1,
                out_shape=(src.height // 4, src.width // 4)
            ).astype(np.float32)

        # Compute metrics
        results = {
            'sample_id': sample_id,
            'water_fraction': compute_water_fraction(mask),
            'edge_density': compute_edge_density(mask),
            **compute_spectral_statistics(bands),
            **compute_ndwi_baseline_performance(bands, mask)
        }

        return results

    except Exception as e:
        print(f"  [WARNING] Error analyzing sample {sample_id}: {e}")
        return None


def analyze_subset(sample_ids: list, subset_name: str, data_dir: Path) -> pd.DataFrame:
    """Analyze a subset of samples."""
    print(f"\nAnalyzing {subset_name} ({len(sample_ids)} samples)...")

    results = []
    for sample_id in tqdm(sample_ids, desc=subset_name):
        result = analyze_sample(sample_id, data_dir)
        if result is not None:
            results.append(result)

    df = pd.DataFrame(results)
    print(f"  Successfully analyzed: {len(results)}/{len(sample_ids)} samples")
    return df


def compute_diversity_metrics(df: pd.DataFrame) -> dict:
    """Compute aggregate diversity metrics."""
    metrics = {
        # Water fraction diversity
        'water_fraction_mean': float(df['water_fraction'].mean()),
        'water_fraction_std': float(df['water_fraction'].std()),
        'water_fraction_range': float(df['water_fraction'].max() - df['water_fraction'].min()),
        'water_fraction_cv': float(df['water_fraction'].std() / (df['water_fraction'].mean() + 1e-8)),

        # Spectral diversity
        'ndwi_mean_std': float(df['ndwi_mean'].std()),
        'ndwi_std_mean': float(df['ndwi_std'].mean()),

        # Scene complexity
        'edge_density_mean': float(df['edge_density'].mean()),
        'edge_density_std': float(df['edge_density'].std()),

        # Difficulty
        'ndwi_f1_mean': float(df['ndwi_f1'].mean()),
        'ndwi_f1_std': float(df['ndwi_f1'].std()),
        'ndwi_f1_min': float(df['ndwi_f1'].min()),
        'ndwi_f1_max': float(df['ndwi_f1'].max()),
    }

    # Composite diversity score
    diversity_score = (
        metrics['water_fraction_cv'] * 0.3 +
        metrics['ndwi_mean_std'] * 0.3 +
        metrics['edge_density_std'] * 0.2 +
        metrics['ndwi_f1_std'] * 0.2
    )
    metrics['diversity_score'] = float(diversity_score)

    return metrics


def create_comparison_plots(df_8: pd.DataFrame, df_45: pd.DataFrame, output_dir: Path):
    """Create visualization comparing subsets."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Dataset Diversity: 8-Sample Subset vs Full 45-Sample Training Set',
                 fontsize=16, fontweight='bold')

    # 1. Water fraction
    ax = axes[0, 0]
    ax.hist(df_8['water_fraction'], bins=8, alpha=0.6, label='8-sample', color='orange', edgecolor='black')
    ax.hist(df_45['water_fraction'], bins=20, alpha=0.6, label='45-sample', color='blue', edgecolor='black')
    ax.set_xlabel('Water Fraction (%)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('(a) Water Fraction Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # 2. NDWI mean
    ax = axes[0, 1]
    ax.hist(df_8['ndwi_mean'], bins=8, alpha=0.6, label='8-sample', color='orange', edgecolor='black')
    ax.hist(df_45['ndwi_mean'], bins=20, alpha=0.6, label='45-sample', color='blue', edgecolor='black')
    ax.set_xlabel('Mean NDWI', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('(b) NDWI Mean Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # 3. NDWI std
    ax = axes[0, 2]
    ax.hist(df_8['ndwi_std'], bins=8, alpha=0.6, label='8-sample', color='orange', edgecolor='black')
    ax.hist(df_45['ndwi_std'], bins=20, alpha=0.6, label='45-sample', color='blue', edgecolor='black')
    ax.set_xlabel('NDWI Std Dev', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('(c) NDWI Within-Sample Variance', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # 4. Edge density
    ax = axes[1, 0]
    ax.hist(df_8['edge_density'], bins=8, alpha=0.6, label='8-sample', color='orange', edgecolor='black')
    ax.hist(df_45['edge_density'], bins=20, alpha=0.6, label='45-sample', color='blue', edgecolor='black')
    ax.set_xlabel('Edge Density (%)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('(d) Scene Complexity', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # 5. NDWI F1 (difficulty)
    ax = axes[1, 1]
    ax.hist(df_8['ndwi_f1'], bins=8, alpha=0.6, label='8-sample', color='orange', edgecolor='black')
    ax.hist(df_45['ndwi_f1'], bins=20, alpha=0.6, label='45-sample', color='blue', edgecolor='black')
    ax.set_xlabel('NDWI Baseline F1 (%)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('(e) Difficulty (NDWI Performance)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # 6. Box plot
    ax = axes[1, 2]
    data_to_plot = [df_8['ndwi_f1'], df_45['ndwi_f1']]
    bp = ax.boxplot(data_to_plot, labels=['8-sample', '45-sample'], patch_artist=True)
    bp['boxes'][0].set_facecolor('orange')
    bp['boxes'][1].set_facecolor('blue')
    ax.set_ylabel('NDWI Baseline F1 (%)', fontsize=12)
    ax.set_title('(f) Difficulty Distribution', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'diversity_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: diversity_comparison.png")


def perform_statistical_tests(df_8: pd.DataFrame, df_45: pd.DataFrame) -> dict:
    """Perform statistical tests."""
    results = {}

    # Kolmogorov-Smirnov tests
    stat, p = stats.ks_2samp(df_8['water_fraction'], df_45['water_fraction'])
    results['water_fraction_ks'] = {'statistic': float(stat), 'p_value': float(p)}

    stat, p = stats.ks_2samp(df_8['ndwi_mean'], df_45['ndwi_mean'])
    results['ndwi_mean_ks'] = {'statistic': float(stat), 'p_value': float(p)}

    stat, p = stats.ks_2samp(df_8['ndwi_f1'], df_45['ndwi_f1'])
    results['ndwi_f1_ks'] = {'statistic': float(stat), 'p_value': float(p)}

    # T-tests
    stat, p = stats.ttest_ind(df_8['ndwi_f1'], df_45['ndwi_f1'])
    results['ndwi_f1_ttest'] = {'statistic': float(stat), 'p_value': float(p)}

    return results


def main():
    """Main execution."""
    print("=" * 80)
    print("STEP 2: QUANTIFYING DATASET DIVERSITY METRICS")
    print("=" * 80)
    print()

    if not DATA_DIR.exists():
        print(f"[ERROR] Dataset not found at {DATA_DIR}")
        return

    # Analyze subsets
    df_8 = analyze_subset(TRAIN_8_SAMPLES, "8-sample ablation subset", DATA_DIR)
    df_45 = analyze_subset(TRAIN_45_SAMPLES, "45-sample full training set", DATA_DIR)

    if df_8.empty or df_45.empty:
        print("[ERROR] Failed to analyze samples")
        return

    # Compute diversity metrics
    print("\n" + "=" * 80)
    print("DIVERSITY METRICS SUMMARY")
    print("=" * 80)

    metrics_8 = compute_diversity_metrics(df_8)
    metrics_45 = compute_diversity_metrics(df_45)

    print("\n8-Sample Subset:")
    for key, value in metrics_8.items():
        print(f"  {key}: {value:.4f}")

    print("\n45-Sample Full Set:")
    for key, value in metrics_45.items():
        print(f"  {key}: {value:.4f}")

    # Statistical tests
    print("\n" + "=" * 80)
    print("STATISTICAL TESTS")
    print("=" * 80)

    stat_tests = perform_statistical_tests(df_8, df_45)
    for test_name, result in stat_tests.items():
        print(f"\n{test_name}:")
        print(f"  Statistic: {result['statistic']:.4f}")
        print(f"  P-value: {result['p_value']:.4f}")
        if result['p_value'] < 0.05:
            print("  Result: Significantly different (p < 0.05)")
        else:
            print("  Result: Not significantly different (p >= 0.05)")

    # Create visualizations
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)
    create_comparison_plots(df_8, df_45, OUTPUT_DIR)

    # Save results
    summary = {
        '8_sample_metrics': metrics_8,
        '45_sample_metrics': metrics_45,
        'statistical_tests': stat_tests,
        'interpretation': {
            'diversity_ratio': metrics_8['diversity_score'] / metrics_45['diversity_score'],
            'difficulty_difference': metrics_8['ndwi_f1_mean'] - metrics_45['ndwi_f1_mean'],
            'assessment': (
                "8-sample subset is EASIER" if metrics_8['ndwi_f1_mean'] > metrics_45['ndwi_f1_mean'] + 2
                else "8-sample subset is SIMILAR difficulty"
            )
        }
    }

    summary_path = OUTPUT_DIR / 'diversity_analysis_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    df_8.to_csv(OUTPUT_DIR / '8_sample_detailed_metrics.csv', index=False)
    df_45.to_csv(OUTPUT_DIR / '45_sample_detailed_metrics.csv', index=False)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Figures: diversity_comparison.png")
    print(f"Summary: diversity_analysis_summary.json")
    print(f"Detailed CSVs: 8_sample_detailed_metrics.csv, 45_sample_detailed_metrics.csv")
    print()
    print("[OK] Diversity analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

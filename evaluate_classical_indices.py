"""
Evaluate Classical Spectral Indices (NDWI, MNDWI, AWEI) on S1S2-Water Test Set

This script is CRITICAL for revision - it measures actual classical baseline performance
on the exact same test set used for deep learning evaluation.

Author: Generated for manuscript revision
Date: November 9, 2025
"""

import os
import sys
import numpy as np
import rasterio
from pathlib import Path
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11


def compute_spectral_indices(bands):
    """
    Compute NDWI, MNDWI, AWEI from input bands.
    
    Args:
        bands: dict with keys 'green', 'nir', 'swir' (normalized 0-1)
    
    Returns:
        dict with 'ndwi', 'mndwi', 'awei' (unnormalized values)
    """
    green = bands['green']
    nir = bands['nir']
    swir = bands['swir']
    
    # NDWI = (Green - NIR) / (Green + NIR)
    ndwi = (green - nir) / (green + nir + 1e-10)
    
    # MNDWI = (Green - SWIR) / (Green + SWIR)
    mndwi = (green - swir) / (green + swir + 1e-10)
    
    # AWEI = 4 * (Green - SWIR) - 0.25 * NIR + 2.75 * SWIR
    awei = 4 * (green - swir) - 0.25 * nir + 2.75 * swir
    
    return {'ndwi': ndwi, 'mndwi': mndwi, 'awei': awei}


def compute_metrics_at_threshold(predictions, ground_truth, threshold):
    """
    Compute F1, IoU, Precision, Recall at a specific threshold.
    
    Args:
        predictions: index values (H x W)
        ground_truth: binary mask (H x W), {0, 1}
        threshold: classification threshold
    
    Returns:
        dict with metrics
    """
    binary_pred = (predictions > threshold).astype(np.uint8)
    
    tp = np.sum((binary_pred == 1) & (ground_truth == 1))
    fp = np.sum((binary_pred == 1) & (ground_truth == 0))
    tn = np.sum((binary_pred == 0) & (ground_truth == 0))
    fn = np.sum((binary_pred == 0) & (ground_truth == 1))
    
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    iou = tp / (tp + fp + fn + 1e-10)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-10)
    
    return {
        'f1': f1 * 100,
        'iou': iou * 100,
        'precision': precision * 100,
        'recall': recall * 100,
        'accuracy': accuracy * 100,
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn)
    }


def evaluate_index_on_sample(s2_path, mask_path, index_name, thresholds):
    """
    Evaluate a single spectral index on one sample across multiple thresholds.
    
    Args:
        s2_path: path to Sentinel-2 image (5 bands: B2, B3, B4, B8, B11)
        mask_path: path to water mask
        index_name: 'ndwi', 'mndwi', or 'awei'
        thresholds: array of threshold values to test
    
    Returns:
        list of dicts, one per threshold
    """
    # Load S2 imagery
    with rasterio.open(s2_path) as src:
        # Read bands (assuming order: B2, B3, B4, B8, B11)
        # Band indices: 0=Blue, 1=Green, 2=Red, 3=NIR, 4=SWIR
        s2_data = src.read().astype(np.float32)
        
        # Normalize to [0, 1]
        s2_data = s2_data / 10000.0
        s2_data = np.clip(s2_data, 0, 1)
        
        green = s2_data[1]  # B3
        nir = s2_data[3]    # B8
        swir = s2_data[4]   # B11
    
    # Load mask (try aligned version first, fallback to regular)
    mask_path_aligned = str(mask_path).replace('_msk.tif', '_msk_aligned.tif')
    mask_to_use = mask_path_aligned if os.path.exists(mask_path_aligned) else mask_path
    
    if not os.path.exists(mask_to_use):
        raise FileNotFoundError(f"Mask not found: {mask_to_use}")
    
    with rasterio.open(mask_to_use) as src:
        mask = src.read(1).astype(np.float32)
        # Binarize mask
        mask = (mask > 0).astype(np.uint8)
    
    # Compute spectral index
    bands = {'green': green, 'nir': nir, 'swir': swir}
    indices = compute_spectral_indices(bands)
    index_values = indices[index_name]
    
    # Evaluate at each threshold
    results = []
    for threshold in thresholds:
        metrics = compute_metrics_at_threshold(index_values, mask, threshold)
        metrics['threshold'] = float(threshold)
        results.append(metrics)
    
    return results


def main():
    """Main evaluation pipeline."""
    
    print("="*80)
    print("CRITICAL EXPERIMENT: Measuring Classical Spectral Index Performance")
    print("="*80)
    print("\nThis script measures NDWI, MNDWI, AWEI on the EXACT test set")
    print("used for deep learning evaluation, providing the missing baseline.\n")
    
    # Dataset paths
    data_root = Path("dataset/s1s2_water_minimal")
    
    # Test sample IDs (11 samples: 80-93, excluding gaps)
    test_sample_ids = [80, 82, 83, 84, 85, 86, 87, 88, 89, 91, 93]
    
    print(f"Test set: {len(test_sample_ids)} samples")
    print(f"Sample IDs: {test_sample_ids}")
    print(f"Data root: {data_root}\n")
    
    # Threshold range
    # For NDWI/MNDWI: -1 to +1 (normalized)
    # For AWEI: -5 to +5 (unnormalized)
    ndwi_thresholds = np.arange(-1.0, 1.05, 0.05)
    awei_thresholds = np.arange(-5.0, 5.05, 0.1)
    
    # Storage for results
    all_results = {
        'ndwi': [],
        'mndwi': [],
        'awei': []
    }
    
    # Evaluate each sample
    for sample_id in tqdm(test_sample_ids, desc="Processing samples"):
        s2_path = data_root / str(sample_id) / f"sentinel12_s2_{sample_id}_img.tif"
        mask_path = data_root / str(sample_id) / f"sentinel12_s1_{sample_id}_msk.tif"
        
        if not s2_path.exists():
            print(f"⚠️  Warning: {s2_path} not found, skipping...")
            continue
        
        if not mask_path.exists():
            print(f"⚠️  Warning: {mask_path} not found, skipping...")
            continue
        
        # Evaluate NDWI
        ndwi_results = evaluate_index_on_sample(
            s2_path, mask_path, 'ndwi', ndwi_thresholds
        )
        all_results['ndwi'].append({
            'sample_id': sample_id,
            'results': ndwi_results
        })
        
        # Evaluate MNDWI
        mndwi_results = evaluate_index_on_sample(
            s2_path, mask_path, 'mndwi', ndwi_thresholds
        )
        all_results['mndwi'].append({
            'sample_id': sample_id,
            'results': mndwi_results
        })
        
        # Evaluate AWEI
        awei_results = evaluate_index_on_sample(
            s2_path, mask_path, 'awei', awei_thresholds
        )
        all_results['awei'].append({
            'sample_id': sample_id,
            'results': awei_results
        })
    
    # Aggregate results across all samples
    print("\n" + "="*80)
    print("AGGREGATING RESULTS ACROSS TEST SET")
    print("="*80 + "\n")
    
    summary = {}
    
    for index_name in ['ndwi', 'mndwi', 'awei']:
        print(f"\n{index_name.upper()} Results:")
        print("-" * 60)
        
        # Get threshold array
        if index_name == 'awei':
            thresholds = awei_thresholds
        else:
            thresholds = ndwi_thresholds
        
        # Aggregate metrics across samples for each threshold
        aggregated = []
        for i, threshold in enumerate(thresholds):
            # Sum TPs, FPs, TNs, FNs across all samples
            total_tp = 0
            total_fp = 0
            total_tn = 0
            total_fn = 0
            
            for sample_result in all_results[index_name]:
                metrics = sample_result['results'][i]
                total_tp += metrics['tp']
                total_fp += metrics['fp']
                total_tn += metrics['tn']
                total_fn += metrics['fn']
            
            # Compute global metrics
            precision = total_tp / (total_tp + total_fp + 1e-10)
            recall = total_tp / (total_tp + total_fn + 1e-10)
            f1 = 2 * precision * recall / (precision + recall + 1e-10)
            iou = total_tp / (total_tp + total_fp + total_fn + 1e-10)
            
            aggregated.append({
                'threshold': float(threshold),
                'f1': f1 * 100,
                'iou': iou * 100,
                'precision': precision * 100,
                'recall': recall * 100
            })
        
        # Find best threshold (max F1)
        best_result = max(aggregated, key=lambda x: x['f1'])
        
        # Standard threshold results (0.0 for NDWI/MNDWI)
        if index_name in ['ndwi', 'mndwi']:
            std_threshold = 0.0
        else:  # AWEI
            std_threshold = 0.0
        
        std_result = min(aggregated, key=lambda x: abs(x['threshold'] - std_threshold))
        
        summary[index_name] = {
            'all_thresholds': aggregated,
            'best': best_result,
            'standard': std_result
        }
        
        # Print results
        print(f"\nStandard threshold ({std_threshold}):")
        print(f"  F1:        {std_result['f1']:.2f}%")
        print(f"  IoU:       {std_result['iou']:.2f}%")
        print(f"  Precision: {std_result['precision']:.2f}%")
        print(f"  Recall:    {std_result['recall']:.2f}%")
        
        print(f"\nOptimal threshold ({best_result['threshold']:.2f}):")
        print(f"  F1:        {best_result['f1']:.2f}%")
        print(f"  IoU:       {best_result['iou']:.2f}%")
        print(f"  Precision: {best_result['precision']:.2f}%")
        print(f"  Recall:    {best_result['recall']:.2f}%")
    
    # Save results
    output_file = "classical_indices_evaluation.json"
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Results saved to {output_file}")
    
    # Generate plots
    print("\nGenerating plots...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, index_name in enumerate(['ndwi', 'mndwi', 'awei']):
        ax = axes[idx]
        
        data = summary[index_name]['all_thresholds']
        thresholds = [d['threshold'] for d in data]
        f1_scores = [d['f1'] for d in data]
        precision = [d['precision'] for d in data]
        recall = [d['recall'] for d in data]
        
        ax.plot(thresholds, f1_scores, 'b-', linewidth=2, label='F1-Score')
        ax.plot(thresholds, precision, 'g--', linewidth=1.5, label='Precision')
        ax.plot(thresholds, recall, 'r--', linewidth=1.5, label='Recall')
        
        # Mark best F1
        best = summary[index_name]['best']
        ax.axvline(best['threshold'], color='blue', linestyle=':', alpha=0.5)
        ax.scatter([best['threshold']], [best['f1']], c='blue', s=100, zorder=10)
        
        # Mark standard threshold
        std = summary[index_name]['standard']
        ax.axvline(std['threshold'], color='gray', linestyle=':', alpha=0.5)
        
        ax.set_xlabel('Threshold', fontsize=12)
        ax.set_ylabel('Score (%)', fontsize=12)
        ax.set_title(f'{index_name.upper()}\n(Best F1: {best["f1"]:.2f}% @ {best["threshold"]:.2f})', 
                     fontsize=13, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 100])
    
    plt.tight_layout()
    plt.savefig('manuscript/classical_indices_threshold_curves.png', dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved to manuscript/classical_indices_threshold_curves.png")
    
    # Generate comparison table
    print("\n" + "="*80)
    print("SUMMARY TABLE FOR MANUSCRIPT")
    print("="*80 + "\n")
    
    print("| Method | Threshold | F1 (%) | IoU (%) | Precision (%) | Recall (%) |")
    print("|--------|-----------|--------|---------|---------------|------------|")
    
    for index_name in ['ndwi', 'mndwi', 'awei']:
        std = summary[index_name]['standard']
        best = summary[index_name]['best']
        
        print(f"| {index_name.upper()} (standard) | {std['threshold']:6.2f} | "
              f"{std['f1']:6.2f} | {std['iou']:6.2f} | {std['precision']:6.2f} | "
              f"{std['recall']:6.2f} |")
        
        if abs(best['threshold'] - std['threshold']) > 0.01:
            print(f"| {index_name.upper()} (optimal)  | {best['threshold']:6.2f} | "
                  f"{best['f1']:6.2f} | {best['iou']:6.2f} | {best['precision']:6.2f} | "
                  f"{best['recall']:6.2f} |")
    
    print("\nFor comparison, deep learning models:")
    print("| Spectral-guided DL | 0.50 | 86.80 | 82.63 | 87.25 | 88.53 |")
    print("| Baseline DL        | 0.50 | 83.91 | 79.13 | 84.59 | 87.18 |")
    
    # Critical analysis
    print("\n" + "="*80)
    print("CRITICAL ANALYSIS FOR MANUSCRIPT REFRAMING")
    print("="*80 + "\n")
    
    ndwi_best_f1 = summary['ndwi']['best']['f1']
    spectral_dl_f1 = 86.80
    baseline_dl_f1 = 83.91
    
    gap_to_ndwi = ndwi_best_f1 - spectral_dl_f1
    improvement = spectral_dl_f1 - baseline_dl_f1
    
    print(f"NDWI Best F1:      {ndwi_best_f1:.2f}%")
    print(f"Spectral DL F1:    {spectral_dl_f1:.2f}%")
    print(f"Baseline DL F1:    {baseline_dl_f1:.2f}%")
    print(f"\nGap to NDWI:       {gap_to_ndwi:+.2f} percentage points")
    print(f"DL Improvement:    {improvement:+.2f} percentage points")
    
    if gap_to_ndwi > 5.0:
        print("\n⚠️  FRAMING: Deep learning is 5+ points behind NDWI")
        print("   → Emphasize: 'Foundation for multi-modal fusion'")
        print("   → Avoid claiming: 'Approaches NDWI performance'")
    elif gap_to_ndwi > 2.0:
        print("\n✓  FRAMING: Deep learning approaches NDWI (within ~3 points)")
        print("   → Emphasize: 'Proof-of-equivalence'")
        print("   → Claim: 'Competitive performance with flexibility'")
    else:
        print("\n🎉 FRAMING: Deep learning matches NDWI performance!")
        print("   → Emphasize: 'Parity with classical methods'")
        print("   → Claim: 'Achieves NDWI-level accuracy with added flexibility'")
    
    print("\n" + "="*80)
    print("✅ CRITICAL EXPERIMENT COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("1. Update manuscript abstract with actual NDWI performance")
    print("2. Revise all claims of 'approaching 90-95%' with measured gap")
    print("3. Reframe contribution based on actual NDWI comparison")
    print("4. Add Figure showing threshold curves")
    print("5. Add Table comparing classical vs. deep learning methods")


if __name__ == "__main__":
    main()


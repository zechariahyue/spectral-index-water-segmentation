# Water Body Segmentation — Reproducibility Code

Code for the paper:
**"Bridging Spectral Indices and Deep Learning: Domain Knowledge Integration for Accurate Water Body Segmentation"**
*Remote Sensing in Earth Systems Sciences*

## Dataset

Download the **S1S2-Water** dataset from Zenodo:
https://zenodo.org/records/11278238

After downloading, place it so the directory structure looks like:

```
code/
  dataset/
    s1s2_water_minimal/
      0/
        s2_B2.tif
        s2_B3.tif
        s2_B4.tif
        s2_B8.tif
        s2_B11.tif
        s2_B12.tif
        water_mask.tif
      1/
        ...
      ...
```

Each numbered folder is one Sentinel-2 scene. The scripts expect the 6 spectral bands (B2, B3, B4, B8, B11, B12) and a binary `water_mask.tif`.

## Requirements

```bash
pip install torch torchvision segmentation-models-pytorch rasterio numpy scikit-learn matplotlib seaborn scipy tqdm
```

Tested with Python 3.10, PyTorch 2.7, CUDA 12.8.

## Scripts

Run in order to reproduce all paper results:

### 1. Classical baselines (NDWI, MNDWI, AWEI)
```bash
python evaluate_classical_indices.py
```
Outputs F1, IoU, precision, recall for all three classical methods on the test set.

### 2. Main result — 8-channel model (Table 1, primary contribution)
```bash
python step3_train_fullscale_ablation.py
```
Trains DeepLabV3+ with RGB+NIR+SWIR+NDWI+MNDWI+AWEI on all 45 training samples.
Expected: **99.17% F1** (no spectral consistency loss).

### 3. AWEI ablation — 7-channel model (Table 2)
```bash
python step4_train_awei_ablation.py
```
Trains without AWEI channel to isolate its contribution.
Expected: **98.36% F1** (−0.81pp vs 8-channel).

### 4. SAR-only baseline (Table 1)
```bash
python step6_train_sar_only.py
```
Trains on Sentinel-1 VV+VH only.
Expected: **0% F1** (S1S2-Water SAR bands are 99% fill values).

### 5. SAR+Optical fusion (Table 1)
```bash
python step7_train_sar_optical_fusion.py
```
Trains 10-channel model (8 optical + VV + VH).
Expected: **93.59% F1**.

### 6. Qualitative figures (Figure 5)
```bash
python step1b_generate_best_results.py
python generate_qualitative_grid.py
```
Generates side-by-side comparisons: RGB input / NDWI / model prediction / ground truth / error map.

### 7. Statistical validation (KS test)
```bash
python step2_quantify_diversity.py
```
Confirms training subset is representative of full distribution (p > 0.05).

## Results Summary

| Configuration | Channels | F1 (%) |
|---|---|---|
| Classical NDWI (θ=0.15) | — | 97.20 |
| **8-channel DeepLabV3+ (ours)** | **8** | **99.17** |
| 7-channel (no AWEI) | 7 | 98.36 |
| 5-channel baseline | 5 | 83.91 |
| 8-channel + consistency loss | 8 | 86.80 |
| SAR+Optical fusion | 10 | 93.59 |
| SAR-only | 2 | 0.00 |

## Citation

```bibtex
@article{zhu2025water,
  title={Bridging Spectral Indices and Deep Learning: Domain Knowledge Integration for Accurate Water Body Segmentation},
  author={Zhu, Yue and Liu, Qingyang},
  journal={Remote Sensing in Earth Systems Sciences},
  year={2025}
}
```

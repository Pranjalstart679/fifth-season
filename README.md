# fifth-season

Spatio-temporal smog forecasting using a multi-task ConvLSTM U-Net, fusing Sentinel-5P satellite imagery with NASA MERRA-2 reanalysis weather data.

---

## Problem

Standard smog forecasting models tend to produce over-smoothed spatial predictions that lose fine-grained edge structure. This project addresses that with a combined MSE + SSIM regression loss, supplemented by a classification head that identifies the dominant aerosol type in each grid cell.

---

## Data Sources

| Dataset | Description | Source |
|---------|-------------|--------|
| **Sentinel-5P** (`S5PL2_5D.nc`) | Atmospheric composition imagery (NO₂, SO₂, aerosol optical depth, etc.) aggregated to 5-day averages | ESA Copernicus |
| **MERRA-2** (`merra-2-till-DEC2025/`) | NASA meteorological reanalysis — wind, temperature, humidity, aerosol column masses; daily files interpolated to match the Sentinel-5P 291×512 grid | NASA GES DISC |

---

## Model Architecture

**Multi-Task Learning U-Net ConvLSTM** (`src/model/convlstm.py`)

```
Input  (B, T=5, H, W, C_features)
  │
  ├── ConvLSTM Encoder  (3 depth levels, return_sequences=True)
  │     └── SE channel attention at each level
  │
  ├── ConvLSTM Bottleneck
  │
  ├── ConvLSTM Decoder  (3 depth levels with U-Net skip connections)
  │
  ├── severity_output  (H, W, 1)  — smog intensity regression
  └── identity_output  (H, W, K)  — dominant aerosol class (K=5)
```

**Losses**

| Head | Loss | Notes |
|------|------|-------|
| Severity | `0.20 × MSE + 0.80 × (1 − SSIM)` | High SSIM weight fights over-smoothing |
| Identity | Sparse focal cross-entropy (γ=2) | Down-weights easy majority-class pixels |

**Metrics tracked:** MSE, Average SSIM (severity); sparse categorical accuracy (identity)

---

## Aerosol Classes

The five MERRA-2 aerosol species compete per grid cell to determine the classification target:

| Index | Variable | Species |
|-------|----------|---------|
| 0 | `BCCMASS` | Black carbon |
| 1 | `DUCMASS` | Dust |
| 2 | `OCCMASS` | Organic carbon |
| 3 | `SO4CMASS` | Sulfate |
| 4 | `SSCMASS` | Sea salt |

The dominant class per cell is determined by the argmax of column mass concentrations.

---

## Data Pipeline

```
Raw Sentinel-5P (.nc)  →  process_sentinel.py  →  sentinel_5d.nc
Raw MERRA-2 (daily .nc) →  process_merra2.py   →  merra2_5d_aligned.nc
                                                          │
                                               fuse_datasets.py
                                                          │
                              ┌───────────────────────────┤
                              ↓                           ↓
                         train_5D.nc               test_5D.nc
                       (up to 2022-12-31)         (year 2023)
```

**Preprocessing steps:**
1. **Sentinel** — variable selection, NaN handling, 5-day temporal aggregation
2. **MERRA-2** — deduplication, hourly → 5-day averaging, bilinear interpolation onto the 291×512 Sentinel grid
3. **Fusion** — inner join on shared timestamps, min-max scaling (statistics fitted on training split only), dominant aerosol target creation
4. **Split** — time-based: train ≤ 2022, test = 2023 (falls back to last available year if 2023 is absent)

---

## Repository Layout

```
fifth-season/
├── src/
│   ├── train.py                  # WeatherDataGenerator + train_model()
│   ├── evaluate.py               # evaluate_model() — MSE, SSIM, accuracy
│   ├── data/
│   │   ├── process_sentinel.py
│   │   ├── process_merra2.py
│   │   └── fuse_datasets.py
│   └── model/
│       ├── convlstm.py           # MTL U-Net ConvLSTM + custom losses/metrics
│       └── training_workflow.ipynb  # Interactive debug/training notebook
├── Dataset/
│   ├── S5PL2_5D.nc
│   ├── merra-2-till-DEC2025/    # Raw daily MERRA-2 NetCDF files
│   └── processed/               # Generated artifacts (gitignored)
├── artifacts/                   # Model checkpoints, training logs (gitignored)
├── environment.yml
├── environment.windows-cuda.yml
└── CUDA_SETUP.md
```

---

## Setup

### CPU / generic
```bash
conda env create -f environment.yml
conda activate fifth-season
```

### Windows + CUDA (RTX GPU)
Follow [CUDA_SETUP.md](CUDA_SETUP.md), then:
```bash
conda env create -f environment.windows-cuda.yml
conda activate fifth-season-cuda
```

Memory growth is enabled automatically in the notebook and training script so TensorFlow does not allocate all VRAM on startup.

---

## Training

### Via notebook (recommended for debugging)
Open `src/model/training_workflow.ipynb` and step through each guarded stage. Toggle the `RUN_*` flags at the top to skip stages whose outputs already exist.

### Via CLI
```bash
# Full training run
python src/train.py

# With custom hyperparameters
python src/train.py \
  --epochs 20 \
  --batch-size 1 \
  --lr 1e-5 \
  --output-dir artifacts/my_run
```

Each training run is saved to a timestamped subdirectory (e.g. `artifacts/run_20260310_143022/`) so experiments are never overwritten. The best checkpoint is also copied to `artifacts/best_model.keras`.

---

## Evaluation

```bash
python src/evaluate.py \
  --model-path artifacts/best_model.keras \
  --test-dataset Dataset/processed/test_5D.nc \
  --output-dir artifacts/evaluation
```

Evaluation produces per-sample MSE, SSIM, and per-class accuracy, plus saved prediction plots.

---

## Key Hyperparameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| Sequence length | 5 | Number of 5-day steps fed as input |
| Batch size | 1 | Full 291×512 spatial frames are memory-intensive |
| Learning rate | 1e-5 | Low LR stabilises ConvLSTM training |
| SSIM weight (α) | 0.80 | Higher = sharper spatial predictions |
| Focal loss γ | 2.0 | Focuses on hard aerosol minority classes |
| Early stopping patience | 4 | Monitors `val_loss` |

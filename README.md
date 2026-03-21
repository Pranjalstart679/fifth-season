# Aerosol Prediction over the IGP (ConvLSTM)

A Deep Learning and Geospatial Data Science project building a Multi-Task ConvLSTM model to predict aerosol index intensities and identify dominant aerosol type compositions (by percentage) as spatio-temporal heatmaps over the Indo-Gangetic Plain (IGP).

## Project Overview
This project fuses high-resolution satellite imagery with coarse global weather reanalysis data to combat the over-smoothing problem common in forecasting models.

**Bounding Box (IGP):** `[68.137, 24.886, 84.836, 34.379]`
**Timeframe:** Jan 1, 2019 – Dec 31, 2025

### Data Sources
1. **Sentinel-5P:** High-resolution daily overpass atmospheric composition.
2. **MERRA-2:** 0.25-degree grid meteorological reanalysis (includes `BCCMASS` Black Carbon, `DUCMASS` Dust, `OCCMASS` Organic Carbon, `SO4CMASS` Sulfate, and `SSCMASS` Sea Salt).

## Repository Layout
Standard geospatial deep learning structure to isolate massive datasets from processing scripts and artifacts:

```
fifth-season/
│
├── data/                          # Data directory (ignored in git)
│   ├── raw/                       # Immutable raw datasets downloaded directly
│   │   ├── merra-2/               # Hourly/Daily .nc files (e.g., merra-2-till-DEC2025)
│   │   └── sentinel-5p/           # High-res, daily overpass S5P files
│   ├── interim/                   # Intermediate data (e.g., temporally subsetted)
│   └── processed/                 # Final, model-ready aligned un-corrupted aggregated data
│
├── notebooks/                     # Jupyter notebooks for interactive EDA
│   ├── 01-eda-merra2.ipynb        # Inspecting MERRA-2 dimensions/vars
│   ├── 02-eda-s5p.ipynb           # Exploring Sentinel-5P data
│   └── 03-spatial-alignment.ipynb # Testing xesmf conservative regridding
│
├── scripts/                       # Modular Python scripts for reproducible execution
│   ├── preprocessing/             # Scripts to clean, merge, and align data
│   │   ├── preprocess_merra2.py   # Merges daily netCDF, temporally resamples to Daily mean
│   │   ├── preprocess_s5p.py      # Sentinel-5P cleaning, bounding box sub-setting
│   │   └── align_spatial.py       # Inter-grid alignment using xesmf
│   │
│   ├── modeling/                  # Model architecture and training scripts
│   │   ├── data_loader.py         # Torch/TF Map Dataset/Dataloader
│   │   ├── convlstm.py            # The Deep Learning Model architecture component
│   │   ├── train.py               # The main model training loop
│   │   └── evaluate.py            # Handles metrics like MSE, SSIM, Categorical Accuracy
│   │
│   └── visualization/             # Heatmaps, comparison figures, plots
│       └── plot_heatmaps.py       # Ground truth vs predicted visualizer
│
├── models/                        # Checkpoints and serialized model states
│   └── checkpoints/
│
├── reports/                       # Evaluation PDFs, PNGs of output heatmaps
│   └── figures/
│
├── requirements.txt               # xarray, dask, xesmf, torch, netCDF4, etc.
└── environment.yml                # Conda requirements depending on OS implementation
```

## Running the Data Pipeline
1. Install requirements: `pip install -r requirements.txt`
2. Ensure you have the C-libraries required by `xarray` installed (`netCDF4`, `h5netcdf`).
3. Run the MERRA-2 pipeline to aggregate daily logs into consolidated averages:
```bash
python scripts/preprocessing/preprocess_merra2.py
```


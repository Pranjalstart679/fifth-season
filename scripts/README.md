# Modular Python Scripts 

This directory contains executable, reproducible Python scripts designed to construct the data pipeline and train the deep learning architectures. It is broken into three logical submodules:

## `preprocessing/`
Handles formatting, cleaning, temporal averaging, and spatial alignment of datasets.
*   `preprocess_merra2.py`: Loops over thousands of raw MERRA-2 daily files, concatenates them along the `time` dimension using chunked `dask` dataframes, resamples their multi-hourly steps (07:30 and 08:30 UTC) into daily means, and exports a unified processed NetCDF.
*   `preprocess_s5p.py`: *(To be completed)* Will filter Sentinel-5P files, remove unneeded variables, mask NaN values, and crop geographical bounds to the Indo-Gangetic Plain.
*   `align_spatial.py`: *(To be completed)* Will execute `xesmf` conservative regridding to structurally project the high-resolution Sentinel-5P data onto the coarser 0.25-degree MERRA-2 grid points.

## `modeling/`
Houses the deep learning architecture and its training regimen.
*   `data_loader.py`: Takes the `data/processed/` outputs and converts them into batch-capable sliding-window map datasets for neural network ingestion.
*   `convlstm.py`: Defines the Multi-Task ConvLSTM implementation.
*   `train.py`: The executable that coordinates model initialization, loss functions, optimizers, and the epoch progression loop.
*   `evaluate.py`: Generates the post-training metrics (MSE, Structural Similarity Index, Sparse Categorical Accuracy).

## `visualization/`
For plotting logic independent of exploratory analysis.
*   `plot_heatmaps.py`: Handles generating ground-truth vs. model-prediction grid overlays and saving them to the `reports/figures/` folder.

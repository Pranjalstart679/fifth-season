# Exploratory Data Analysis (EDA) Notebooks

These interactive Jupyter Notebooks are utilized for experimenting, plotting distributions, and understanding the shape, variance, and anomalies of the data before we formalize the actions into the `scripts/` pipeline.

## Notebook Manifest
- `01-eda-merra2.ipynb`: *(To be created)* Will be used to dissect a single `.nc` MERRA-2 file, understand its internal structure, visualize the specific aerosol types (`BCCMASS`, `DUCMASS`, etc.), and verify spatial bounds mapping over the Indo-Gangetic Plain.
- `02-eda-s5p.ipynb`: *(To be created)* Will inspect Sentinel-5P data density, deal with cloud masking, and check overpass timestamps.
- `03-spatial-alignment.ipynb`: *(To be created)* A sandbox to visually test the math behind `xesmf` conservative regridding, to ensure no mass representations are lost when downscaling Sentinel grids to match MERRA grids.

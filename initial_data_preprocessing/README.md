# MERRA-2 Data Preprocessing Pipeline

This folder contains the Python scripts required to clean, process, and temporally align NASA's MERRA-2 aerosol data with ESA's Sentinel-5P 5-day composites.

## 📌 The Goal
Raw MERRA-2 data is downloaded as thousands of individual daily `.nc` files containing sub-daily (hourly) timestamps. Sentinel-5P data, however, is provided as a 5-day temporally averaged composite. 

To train a machine learning model, the input features (MERRA-2) must speak the exact same temporal language as the target labels (Sentinel-5P). This pipeline automates that conversion.

## 🛠️ Prerequisites
You will need to install the dependencies, from requirements.txt
📂 Execution Order
The scripts must be run in the following chronological order. (Note: Update the absolute file paths inside each script to point to your local data directory before running).

01_checkDates.py
Scans the raw downloaded MERRA-2 .nc files to ensure there are no missing days in the multi-year timeline.

02_delCopies.py
Cleans the raw data directory by identifying and removing any accidental duplicate file downloads (e.g., files ending in (1).nc).

03_merra_2.py
Action: Lazily loads all 2,500+ raw daily files into a single massive data cube.

Math: Averages the specific UTC target hours (07:30 and 08:30) into a single mathematically perfect daily mean.

Output: Relabels the daily map to 00:00:00 and saves it as MERRA2_Combined_Daily_Means.nc.

04_merra_5day_gap.py
Action: Takes the combined daily dataset and applies a left-closed temporal resampling (time='5D').

Math: Averages 5 days of data at a time to perfectly match the Sentinel-5P temporal resolution, automatically handling leap years and month-ends.

Output: Applies lossless zlib compression (complevel=4) and saves the final temporal dataset as MERRA2_5Day_Averages.nc.

Note: The final step of the pipeline, Spatial Regridding, is currently in development.
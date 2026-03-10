import xarray as xr
import os
import glob
from pathlib import Path

# --- User Inputs ---
# (Using the paths from our previous steps)
cleaned_merra_folder = r"C:\Users\Swagata\Downloads\merra-2-till-DEC2025"
output_daily_file = r"C:\Users\Swagata\Downloads\MERRA2_Combined_Daily_Means.nc"

def combine_daily_timestamps(source_folder, save_path):
    # 1. Find all clean .nc files in your folder
    file_pattern = os.path.join(source_folder, "*.nc")
    file_list = sorted(glob.glob(file_pattern)) # sorted ensures correct chronological loading
    
    if not file_list:
        print("❌ Error: No .nc files found in the source folder.")
        return

    print(f"Found {len(file_list)} daily files.")
    print("Opening all files at once (using lazy loading)...")

    # 2. Use open_mfdataset to lazily load all files into a single object.
    # combine='by_coords' uses the date in the files to order them correctly.
    # parallel=True (requires dask installed) can speed this up significantly if available.
    try:
        ds = xr.open_mfdataset(file_list, combine='by_coords')
    except ImportError:
        # Fallback if dask isn't installed for parallel processing
        print("Dask not found. Loading files sequentially (this will be slower).")
        ds = xr.open_mfdataset(file_list, combine='by_coords', parallel=False)

    print("\n--- Original Dataset Structure ---")
    print(ds)
    # Check your 'time' dimension here. If you had 100 days of data, 
    # it should now say time=200, because it has both 07:30 and 08:30 for every day.

    # 3. Calculate the mathematical mean of the daily times.
    # .resample(time='1D') groups the data into full days.
    # .mean() calculates the average of the 7:30 and 8:30 arrays.
    print("\nCalculating daily means for the entire timeline...")
    daily_ds = ds.resample(time='1D').mean()

    print("\n--- New Daily Averaged Structure ---")
    print(daily_ds)
    # Check your 'time' dimension now. It should have halved (e.g., time=100).
    # You are left with exactly "one picture per day."

    # 4. Save this massive combined file.
    # Using simple compression (zlib) is highly recommended for massive dataset cubes.
    print(f"\nSaving the combined daily 'data cube' to:\n{save_path}")
    print("This step performs the actual calculation and saving. It might take a few minutes...")
    
    comp = dict(zlib=True, complevel=4) # Light zlib compression
    encoding = {var: comp for var in daily_ds.data_vars}
    
    daily_ds.to_netcdf(save_path, encoding=encoding)

    # Clean up (release memory)
    ds.close()
    daily_ds.close()
    
    print("\n✅ Complete! You now have a single file with one picture per day.")

# Run the Step 2 function
combine_daily_timestamps(cleaned_merra_folder, output_daily_file)
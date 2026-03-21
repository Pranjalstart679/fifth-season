import os
import glob
import xarray as xr

# ==============================================================
# Configuration
# ==============================================================
# Directory containing your daily MERRA-2 NetCDF files
DATA_DIR = r"D:\Minor Projects\Sem-4\data\raw\merra-2\merra-2-till-DEC2025" 
FILE_PATTERN = "*.nc" # Or "*.nc4" depending on your actual file extensions
OUTPUT_FILE_PATH = r"D:\Minor Projects\Sem-4\data\processed\MERRA2_Daily_IGP_Aligned.nc"

def process_merra2_data(data_dir, file_pattern, output_path):
    print(f"--- Starting MERRA-2 Processing Pipeline ---")
    
    # 1. Discover files
    search_path = os.path.join(data_dir, file_pattern)
    file_paths = glob.glob(search_path)
    file_paths.sort() # Ensure chronological order
    
    if not file_paths:
        print(f"Error: No files found matching '{search_path}'")
        return None
        
    print(f"Found {len(file_paths)} files to process.")
    
    try:
        # 2. Load and Merge
        print("Merging datasets along the time dimension...")
        # Use open_mfdataset to lazily load and concatenate over 'time'
        ds = xr.open_mfdataset(
            file_paths,
            combine="by_coords", # Merges based on coordinate values
            chunks="auto",       # Essential for large out-of-core operations
            parallel=True,       # Uses Dask to load files in parallel
            engine="netcdf4"
        )
        
        print("\n--- Pre-Resample Time Dimension ---")
        print(f"Original Time Steps: {ds.sizes['time']}")
        
        # 3. Temporal Averaging
        # Resample the 07:30 and 08:30 UTC time steps into a single daily average
        print("\nResampling time to daily frequency ('1D') and calculating mean...")
        ds_daily = ds.resample(time="1D").mean()
        
        print("\n--- Post-Resample Time Dimension ---")
        print(f"Resampled Time Steps (Days): {ds_daily.sizes['time']}")
        
        # 4. Save to Disk
        print(f"\nSaving daily averaged dataset to: {output_path}")
        ds_daily.to_netcdf(output_path)
        print("Successfully saved MERRA2_Daily_IGP_Aligned.nc!")
        
        return ds_daily
        
    except Exception as e:
        print(f"An error occurred during processing: {e}")
        return None

if __name__ == "__main__":
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(OUTPUT_FILE_PATH), exist_ok=True)
    
    # Run the full pipeline
    process_merra2_data(DATA_DIR, FILE_PATTERN, OUTPUT_FILE_PATH)


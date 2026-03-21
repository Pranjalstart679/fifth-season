import xarray as xr
import os

# ==============================================================
# Configuration
# ==============================================================

S5P_FILE_PATH = r"D:\Minor Projects\Sem-4\data\raw\sentinel-5p\S5PL2_5D.nc"

def inspect_s5p(filepath):
    print(f"--- Inspecting Sentinel-5P File: {os.path.basename(filepath)} ---")
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        print("Please ensure the 1.8GB S5PL2_5D.nc file is placed in the correct directory.")
        return
        
    try:
       
        ds = xr.open_dataset(filepath)
        print(ds)
        
        print("\n--- Dimensions ---")
        print(ds.sizes)
        
        print("\n--- Data Variables ---")
        for var in ds.data_vars:
            print(f"- {var}: {ds[var].dtype}, dims: {ds[var].dims}")
            
        ds.close()
    except Exception as e:
        print(f"Failed to inspect the dataset: {e}")

if __name__ == "__main__":
    inspect_s5p(S5P_FILE_PATH)


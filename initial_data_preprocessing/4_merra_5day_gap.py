import xarray as xr

# --- User Inputs ---
# We use the file we just created as the input
daily_file_path = r"C:\Users\Swagata\Downloads\MERRA2_Combined_Daily_Means.nc"
# We will save the final temporal dataset here
output_5d_file = r"C:\Users\Swagata\Downloads\MERRA2_5Day_Averages.nc"

def create_5day_composites(input_file, output_file):
    print(f"Loading daily dataset from: {input_file}")
    # We only need open_dataset now, since it is all in one file!
    ds = xr.open_dataset(input_file)
    
    print("\n--- Current Daily Structure ---")
    print(f"Number of days: {ds.sizes['time']}")
    
    print("\nCalculating 5-day averages...")
    
    # The crucial resampling step:
    # '5D' groups into 5-day chunks starting from the first date.
    # closed='left' ensures it includes Days 1-5, but strictly excludes Day 6.
    # label='left' ensures the new averaged map is named after Day 1 (matching Sentinel).
    ds_5d = ds.resample(time='5D', closed='left', label='left').mean()
    
    print("\n--- New 5-Day Structure ---")
    print(f"Number of 5-day chunks: {ds_5d.sizes['time']}")
    print("\nFirst three timestamps to verify alignment:")
    # This prints out the first three dates so you can physically see them match Sentinel
    for t in ds_5d.time.values[:3]:
        print(f" - {t}")
    
    print(f"\nSaving the 5-Day composited data cube to:\n{output_file}")
    print("Saving...")
    
    # Adding compression so the final file is lightweight and easy to load into your ML model
    comp = dict(zlib=True, complevel=4)
    encoding = {var: comp for var in ds_5d.data_vars if var != 'time'}
    
    ds_5d.to_netcdf(output_file, encoding=encoding)
    
    # Free up computer memory
    ds.close()
    ds_5d.close()
    
    print("\n✅ Success! Your MERRA-2 dataset is now temporally aligned with Sentinel-5P.")

# Run the Step 3 function
create_5day_composites(daily_file_path, output_5d_file)
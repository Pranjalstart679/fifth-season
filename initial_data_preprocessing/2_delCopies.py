import os
import re
from pathlib import Path

# Set the path to your clean folder
folder_path = r"C:\Users\Swagata\Downloads\merra-2-till-DEC2025" 

def clean_duplicate_files(directory, dry_run=True):
    path = Path(directory)
    
    # This regex pattern looks for Windows duplicate markers at the end of the .nc file
    # It catches things like "filename (1).nc", "filename(2).nc", or "filename - Copy.nc"
    duplicate_pattern = re.compile(r'(\(\d+\)| - Copy)\.nc$')
    
    count = 0
    print(f"Scanning folder: {directory}...\n")
    
    # Loop through every .nc file in the folder
    for file_path in path.glob('*.nc'):
        if duplicate_pattern.search(file_path.name):
            if dry_run:
                print(f"[SAFE MODE] Would delete: {file_path.name}")
            else:
                os.remove(file_path)
                print(f"Deleted: {file_path.name}")
            count += 1
            
    print("\n--- Summary ---")
    if dry_run:
        print(f"Found {count} duplicate files.")
        print("Safety is ON. No files were actually deleted.")
        print("To delete them, change `dry_run=False` and run again.")
    else:
        print(f"Successfully deleted {count} duplicate files. Your folder is clean!")

# Run the function (Leave it as True for your first test!)
clean_duplicate_files(folder_path, dry_run=False)
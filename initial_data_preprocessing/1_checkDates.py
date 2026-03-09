import re
from datetime import datetime, timedelta
from pathlib import Path

# Your clean folder path
folder_path = r"C:\Users\Swagata\Downloads\merra-2-till-DEC2025" 

def check_missing_dates(directory):
    path = Path(directory)
    
    # This regex looks for exactly 8 digits surrounded by dots (e.g., .20190101.)
    date_pattern = re.compile(r'\.(\d{8})\.')
    
    found_dates = set()
    
    print(f"Scanning folder: {directory}...\n")
    
    # 1. Extract every date from the filenames
    for file_path in path.glob('*.nc'):
        match = date_pattern.search(file_path.name)
        if match:
            date_str = match.group(1)
            # Convert the string into an actual Date object
            file_date = datetime.strptime(date_str, "%Y%m%d").date()
            found_dates.add(file_date)
            
    if not found_dates:
        print("Could not find any dates. Please check the folder path and filenames.")
        return

    # 2. Find the absolute start and end of your dataset
    start_date = min(found_dates)
    end_date = max(found_dates)
    
    print(f"Earliest file: {start_date}")
    print(f"Latest file:   {end_date}\n")
    
    # 3. Create a list of every single day that *should* exist between start and end
    expected_dates = set()
    current_date = start_date
    while current_date <= end_date:
        expected_dates.add(current_date)
        current_date += timedelta(days=1)
        
    # 4. Compare what should exist against what actually exists
    missing_dates = expected_dates - found_dates
    
    if not missing_dates:
        print("✅ Excellent! Your files are perfectly continuous. No missing dates.")
    else:
        print(f"❌ Warning: Found {len(missing_dates)} missing dates!")
        for missing in sorted(missing_dates):
            print(f"Missing: {missing.strftime('%Y-%m-%d')}")

# Run the check
check_missing_dates(folder_path)
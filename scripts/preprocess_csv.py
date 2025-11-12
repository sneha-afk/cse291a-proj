import pandas as pd
import os
from pathlib import Path
from datetime import datetime

def convert_csv_dates_to_iso(input_folder: str = "dataset"):
    """
    Recursively convert dates in CSV files to ISO format and rename files.
    Converts dates like 10/4/2020 to 2020-10-04
    """
    folder_path = Path(input_folder)
    csv_files = list(folder_path.rglob("*.csv"))
    
    print(f"Found {len(csv_files)} CSV files to process")
    
    for csv_file in csv_files:
        try:
            # Read the CSV file
            df = pd.read_csv(csv_file)
            
            # Check if 'Date' column exists
            if 'Date' in df.columns:
                # Convert dates to ISO format
                def convert_date(date_str):
                    try:
                        # Parse various date formats
                        if '/' in date_str:
                            # Handle MM/DD/YYYY format
                            return datetime.strptime(date_str, '%m/%d/%Y').strftime('%Y-%m-%d')
                        elif '-' in date_str:
                            # Handle other formats if needed
                            return date_str
                        else:
                            return date_str  # Return as-is if format not recognized
                    except (ValueError, TypeError):
                        return date_str  # Return original if conversion fails
                
                df['Date'] = df['Date'].apply(convert_date)
            
            # Create new filename
            parent_dir = csv_file.parent
            new_filename = f"proc_{csv_file.name}"
            new_file_path = parent_dir / new_filename
            
            # Save with ISO format dates
            df.to_csv(new_file_path, index=False)
            
            print(f"Processed: {csv_file} -> {new_file_path}")
            
        except Exception as e:
            print(f"Error processing {csv_file}: {str(e)}")

if __name__ == "__main__":
    convert_csv_dates_to_iso("dataset")
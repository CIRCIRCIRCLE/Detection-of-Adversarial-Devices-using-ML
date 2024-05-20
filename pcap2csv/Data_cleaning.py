import os
import pandas as pd
from tqdm import tqdm

def clean_csv_files(directory):
    csv_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.csv')]
    
    # Iterate through each csv file
    for csv_file in tqdm(csv_files, desc="Cleaning CSV files"):
        try:
            df = pd.read_csv(csv_file)
            # Drop rows with any NaN values
            cleaned_df = df.dropna()
            # cleaned_df['label'] = 'Benign'
            
            # Save the cleaned dataframe back to the csv file
            cleaned_df.to_csv(csv_file, index=False)
            print(f"Cleaned {csv_file}, removed {len(df) - len(cleaned_df)} rows.")
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")

if __name__ == "__main__":
    # Directory containing CSV files
    csv_directory = "../raw_data/IoT-23/csv_files/"
    
    # Perform the cleaning operation
    clean_csv_files(csv_directory)

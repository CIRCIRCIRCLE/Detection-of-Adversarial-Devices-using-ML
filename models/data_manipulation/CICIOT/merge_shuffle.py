import pandas as pd
from sklearn.utils import shuffle
import os

def read_pkl(pkl_path):
    return pd.read_pickle(pkl_path)

def merge_and_shuffle_pkl_files(pkl1_path, pkl2_path, output_csv_path):
    # Read the PKL files into DataFrames
    df1 = read_pkl(pkl1_path)
    df2 = read_pkl(pkl2_path)
    
    # Concatenate the DataFrames
    merged_df = pd.concat([df1, df2], ignore_index=True)
    
    # Shuffle the DataFrame
    shuffled_df = shuffle(merged_df)
    
    # Save the shuffled DataFrame to a new CSV file
    shuffled_df.to_csv(output_csv_path, index=False)

if __name__ == "__main__":
    current_directory  = os.path.dirname(__file__) 

    # Paths to the input PKL files
    pkl1_path = os.path.join(current_directory, '..', '..', '..','datasets', 'TON_attack.pkl')
    pkl2_path = os.path.join(current_directory, '..', '..', '..','datasets', 'TON_normal.pkl')
    
    # Path to the output merged and shuffled CSV file
    output_csv_path = os.path.join(current_directory, '..', '..', '..','datasets', 'TON_formatted.csv')
    
    # Merge and shuffle the PKL files, then save to a CSV file
    merge_and_shuffle_pkl_files(pkl1_path, pkl2_path, output_csv_path)
    
    print(f"Merged and shuffled dataset saved to {output_csv_path}")

"""
Modified script from the CIC IOT dataset pcacptocsv.

Modified to work on a single thread to convert our own datasets.

Additionally errors were found running the script on Kali Linux 
2024.1. Hence some lines were updated to correctly convert it.

Original Code: https://www.unb.ca/cic/datasets/index.html

Modified by: Yuanyuan Zhou
"""

import os
import time
import warnings
import pandas as pd
from tqdm import tqdm
from Feature_extraction import Feature_extraction

warnings.filterwarnings("ignore")

def split_pcap_file(pcap_file, split_directory, subfiles_size):
    os.system(
        f"tcpdump -r {pcap_file} -w {split_directory}split_temp -C {subfiles_size}"
    )

def convert_pcap_to_csv(subfile, split_directory, destination_directory):
    fe = Feature_extraction()
    subpcap_file = os.path.join(split_directory, subfile)
    try:
        fe.pcap_evaluation(subpcap_file, os.path.join(destination_directory, subfile.split(".")[0] + '.csv'))
    except Exception as e:
        print(f"Error converting {subpcap_file}: {e}")

def merge_csv_files(csv_subfiles, destination_directory, output_file, label):
    mode = "w"
    for csv_file in csv_subfiles:
        try:
            df = pd.read_csv(os.path.join(destination_directory, csv_file))
            df['label'] = label  # Add the label column
            df.to_csv(output_file, header=(mode == "w"), index=False, mode=mode)
            mode = "a"
        except Exception as e:
            print(f"Error merging {csv_file}: {e}")

def remove_files(file_list, directory):
    for file in file_list:
        file_path = os.path.join(directory, file)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Error removing {file_path}: {e}")

if __name__ == "__main__":
    start = time.time()
    print("========== CIC IoT feature extraction ==========")

    # Directory containing PCAP files
    pcap_directory = "../raw_data/TON_IOT/"
    label = "Malicious"
    subfiles_size = 10  # MB
    split_directory = "split_temp/"
    destination_directory = "output/"
    converted_csv_files_directory = "../raw_data/TON_IOT/csv_files/"

    # Ensure the output directory exists
    os.makedirs(converted_csv_files_directory, exist_ok=True)

    # List all pcap files in the directory
    pcapfiles = [os.path.join(pcap_directory, f) for f in os.listdir(pcap_directory) if f.endswith('.pcap')]

    for pcap_file in pcapfiles:
        lstart = time.time()
        print(f"Processing file: {pcap_file}")

        try:
            # Step 1: Split the pcap file
            print(">>>> 1. Splitting the .pcap file.")
            split_pcap_file(pcap_file, split_directory, subfiles_size)

            subfiles = os.listdir(split_directory)
            print(f"Split into {len(subfiles)} subfiles.")

            if not subfiles:
                print(f"No subfiles created for {pcap_file}, skipping to next file.")
                continue
            
            # Step 2: Convert the split pcap files to csv
            print(">>>> 2. Converting (sub) .pcap files to .csv files.")
            for subfile in tqdm(subfiles, desc="Converting to CSV"):
                convert_pcap_to_csv(subfile, split_directory, destination_directory)

            # Step 3: Remove the split pcap files
            print(">>>> 3. Removing (sub) .pcap files.")
            remove_files(subfiles, split_directory)

            # Step 4: Merge the csv files and save to the specified directory
            print(">>>> 4. Merging (sub) .csv files (summary).")
            csv_subfiles = os.listdir(destination_directory)
            if not csv_subfiles:
                print(f"No CSV files created for {pcap_file}, skipping to next file.")
                continue

            output_csv_file = os.path.join(converted_csv_files_directory, os.path.basename(pcap_file) + ".csv")
            merge_csv_files(csv_subfiles, destination_directory, output_csv_file, label)

            # Step 5: Remove the generated csv files
            print(">>>> 5. Removing (sub) .csv files.")
            remove_files(csv_subfiles, destination_directory)

            print(f"Done! ({pcap_file}) ({str(round(time.time() - lstart, 2))}s)")
        
        except Exception as e:
            print(f"Error processing file {pcap_file}: {e}")

    end = time.time()
    print(f"Elapsed Time = {(end-start)}s")

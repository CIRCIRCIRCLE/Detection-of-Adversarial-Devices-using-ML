import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import os
import pickle

'''
The original size of dataset is over 15GB
Functions:
    change the data type  -->5.3+GB
    keep the 22 important features
    aggregate the whole dataset into one .pkl file
'''

def convert_dtype(df):
    #adjust data type
    for col, typ in dtypes.items():
        df[col] = df[col].astype(typ)

    #format columns names to lowercase
    df.columns = df.columns.str.lower().str.replace(' ','_')
    return df

def concat_dfs(files, path):
    cnt = 0
    for file in tqdm(files):
        file_path = os.path.join(path, file)
        if cnt == 0:
            df = pd.read_csv(file_path)
            convert_dtype(df)
        else:
            df_new = pd.read_csv(file_path)
            convert_dtype(df_new)
            df = pd.concat([df, df_new], ignore_index=True)
        cnt += 1
    return df


current_directory  = os.path.dirname(__file__)   #\code
path = os.path.join(current_directory, '..', '..', 'CICIoT2023')

# Find all CSV files in the dataset directory and sort them
df_sets = [k for k in os.listdir(path) if k.endswith('.csv')]
df_sets.sort()
# Add percent for portion test
df_sets = df_sets[:int(len(df_sets)*.2)]

X_columns = [
    'flow_duration', 'Header_Length', 'Protocol Type', 'Duration','Rate', 'Srate', 'Drate', 'fin_flag_number', 'syn_flag_number',
    'rst_flag_number', 'psh_flag_number', 'ack_flag_number','ece_flag_number', 'cwr_flag_number', 'ack_count','syn_count', 'fin_count', 'urg_count', 'rst_count', 
    'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC', 'TCP', 'UDP', 'DHCP', 'ARP', 'ICMP', 'IPv', 'LLC', 'Tot sum', 'Min',
    'Max', 'AVG', 'Std', 'Tot size', 'IAT', 'Number', 'Magnitue', 'Radius', 'Covariance', 'Variance', 'Weight' 
]
y_column = 'label'

dtypes = {
        'flow_duration': np.float32, 'Header_Length': np.uint32, 'Protocol Type': str, 'Duration': np.float32, 'Rate': np.uint32, 'Srate': np.uint32, 'Drate': np.float32, 
        'fin_flag_number': np.bool_, 'syn_flag_number': np.bool_, 'rst_flag_number': np.bool_, 'psh_flag_number': np.bool_, 'ack_flag_number': np.bool_, 'ece_flag_number': np.bool_, 'cwr_flag_number': np.bool_,
        'ack_count': np.float16, 'syn_count': np.float16, 'fin_count': np.uint16, 'urg_count': np.uint16, 'rst_count': np.uint16, 
        'HTTP': np.bool_, 'HTTPS': np.bool_, 'DNS': np.bool_, 'Telnet': np.bool_, 'SMTP': np.bool_, 'SSH': np.bool_, 'IRC': np.bool_, 'TCP': np.bool_, 'UDP': np.bool_, 
        'DHCP': np.bool_, 'ARP': np.bool_, 'ICMP': np.bool_, 'IPv': np.bool_, 'LLC': np.bool_,
        'Tot sum': np.float32, 'Min': np.float32, 'Max': np.float32, 'AVG': np.float32, 'Std': np.float32, 'Tot size': np.float32, 'IAT': np.float32, 'Number': np.float32,
        'Magnitue': np.float32, 'Radius': np.float32, 'Covariance': np.float32, 'Variance': np.float32, 'Weight': np.float32
}

important_features =  ['flow_duration', 'duration', 'srate', 'drate', 'syn_flag_number', 
                       'psh_flag_number', 'ack_flag_number', 'ack_count', 'syn_count', 'rst_count', 
                       'header_length', 'https', 'ssh', 'tot_sum', 'min', 'max', 'avg', 'iat', 
                       'magnitue', 'radius', 'variance', 'weight', 'label']


#concatenate dataframes and save to a pickle file
df = concat_dfs(df_sets, path)[important_features]
output_file = os.path.join(current_directory, '..', '..', 'datasets', 'CICIoT2023_test.pkl')
df.to_pickle(output_file)

#df1 = pd.read_pickle(output_file)
#print(df1.info())
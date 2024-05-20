import os
import pandas as pd
import numpy as np

'''
sample data from the .pkl file for further processing

Chooses some percentage of the dataframe's rows at random while the class balance is preserved. 
map original 34 classes into 8 classes and 2 classes
store the datasets df34, df7, df2 in .csv format
'''


def sample_rows(df, percent_rows):
    
    labels = df['label'].unique()
    dfs_condensed = []
    
    # Select rows with chosen label
    for label in labels:
        mask = df['label'] == label
        df_by_label = df[mask]
        
        # Randomly sample some percentage of rows in current class
        sample = df_by_label.sample(frac = percent_rows)
        dfs_condensed.append(sample)
    
    # Shuffle all samples
    return pd.concat(dfs_condensed, ignore_index = True).sample(frac = 1)


current_directory  = os.path.dirname(__file__)   
dataset_path = os.path.join(current_directory, '..', '..', 'datasets', 'CICIoT2023_test.pkl')
# print(dataset_path)


data = pd.read_pickle(dataset_path)  #5.3+GB
df34 = sample_rows(data, percent_rows=0.5)


print('Unique values in y_column: {}'.format(len(df34['label'].unique())))
print(df34['label'].unique())
# attack_labels = df34['label'].unique()

dict_8_classes = { 'BenignTraffic': 'Benign',   
                    'DDoS-RSTFINFlood': 'DDoS', 'DDoS-PSHACK_Flood': 'DDoS', 'DDoS-SYN_Flood': 'DDoS', 
                    'DDoS-UDP_Flood': 'DDoS', 'DDoS-TCP_Flood': 'DDoS', 'DDoS-ICMP_Flood': 'DDoS', 
                    'DDoS-SynonymousIP_Flood': 'DDoS', 'DDoS-ACK_Fragmentation': 'DDoS', 
                    'DDoS-UDP_Fragmentation': 'DDoS', 'DDoS-ICMP_Fragmentation': 'DDoS', 
                    'DDoS-SlowLoris': 'DDoS', 'DDoS-HTTP_Flood': 'DDoS', 'DoS-UDP_Flood': 'DoS', 
                    'DoS-SYN_Flood': 'DoS', 'DoS-TCP_Flood': 'DoS', 'DoS-HTTP_Flood': 'DoS', 
                    'Mirai-greeth_flood': 'Mirai', 'Mirai-greip_flood': 'Mirai', 'Mirai-udpplain': 'Mirai', 
                    'Recon-PingSweep': 'Recon', 'Recon-OSScan': 'Recon', 'Recon-PortScan': 'Recon', 
                    'VulnerabilityScan': 'Recon', 'Recon-HostDiscovery': 'Recon', 'DNS_Spoofing': 'Spoofing', 
                    'MITM-ArpSpoofing': 'Spoofing', 'BrowserHijacking': 'Web', 
                    'Backdoor_Malware': 'Web', 'XSS': 'Web', 'Uploading_Attack': 'Web', 'SqlInjection': 'Web', 
                    'CommandInjection': 'Web', 'DictionaryBruteForce': 'BruteForce'}                                                                                                            
                                                                                                                                 

dict_2_classes = { 'BenignTraffic': 'Benign',   
                    'DDoS-RSTFINFlood': 'Malicious', 'DDoS-PSHACK_Flood': 'Malicious', 'DDoS-SYN_Flood': 'Malicious', 
                    'DDoS-UDP_Flood': 'Malicious', 'DDoS-TCP_Flood': 'Malicious', 'DDoS-ICMP_Flood': 'Malicious', 
                    'DDoS-SynonymousIP_Flood': 'Malicious', 'DDoS-ACK_Fragmentation': 'Malicious', 
                    'DDoS-UDP_Fragmentation': 'Malicious', 'DDoS-ICMP_Fragmentation': 'Malicious', 
                    'DDoS-SlowLoris': 'Malicious', 'DDoS-HTTP_Flood': 'Malicious', 'DoS-UDP_Flood': 'Malicious', 
                    'DoS-SYN_Flood': 'Malicious', 'DoS-TCP_Flood': 'Malicious', 'DoS-HTTP_Flood': 'Malicious', 
                    'Mirai-greeth_flood': 'Malicious', 'Mirai-greip_flood': 'Malicious', 'Mirai-udpplain': 'Malicious', 
                    'Recon-PingSweep': 'Malicious', 'Recon-OSScan': 'Malicious', 'Recon-PortScan': 'Malicious', 
                    'VulnerabilityScan': 'Malicious', 'Recon-HostDiscovery': 'Malicious', 'DNS_Spoofing': 'Malicious', 
                    'MITM-ArpSpoofing': 'Malicious', 'BrowserHijacking': 'Malicious', 
                    'Backdoor_Malware': 'Malicious', 'XSS': 'Malicious', 'Uploading_Attack': 'Malicious', 'SqlInjection': 'Malicious', 
                    'CommandInjection': 'Malicious', 'DictionaryBruteForce': 'Malicious'}   
df8 = df34.copy()
df2 = df34.copy()
df8['label'] = df8['label'].map(dict_8_classes)
df2['label'] = df2['label'].map(dict_2_classes)


df34.to_csv(os.path.join(current_directory, '..', '..', 'datasets', 'df34.csv'), index=False)
df8.to_csv(os.path.join(current_directory, '..', '..', 'datasets', 'df8.csv'), index=False)
df2.to_csv(os.path.join(current_directory, '..', '..', 'datasets', 'df2.csv'), index=False)

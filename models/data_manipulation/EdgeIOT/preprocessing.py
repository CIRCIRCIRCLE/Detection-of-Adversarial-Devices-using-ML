import pandas as pd
import numpy as np
import os
from sklearn.utils import shuffle

current_directory  = os.path.dirname(__file__)  
dataset_path = os.path.join(current_directory, '..', '..', '..', 'datasets', 'EdgeIIoT.csv')

df = pd.read_csv(dataset_path)

# Step1: To save memory, convert float64 type into float16
float64_columns = df.select_dtypes(include=['float64']).columns
df[float64_columns] = df[float64_columns].astype('float16')



# Step2: Drop redundant cols and clean missing values
drop_columns = ["frame.time", "ip.src_host", "ip.dst_host", "arp.src.proto_ipv4","arp.dst.proto_ipv4", 
                "http.file_data","http.request.full_uri","icmp.transmit_timestamp",
                "http.request.uri.query", "tcp.options","tcp.payload","tcp.srcport",
                "tcp.dstport", "udp.port", "mqtt.msg"]

df.drop(drop_columns, axis=1, inplace=True)
df.dropna(axis=0, how='any', inplace=True)
df.drop_duplicates(subset=None, keep="first", inplace=True)

df = shuffle(df)
df.isna().sum()
print(df['Attack_type'].value_counts())
# print(df.head())
print(df.info())  #47features

# Step3: Encode the object type
def encode_text_dummy(df, name):

    dummies = pd.get_dummies(df[name])

    for x in dummies.columns:

        dummy_name = f"{name}-{x}"

        df[dummy_name] = dummies[x]

    df.drop(name, axis=1, inplace=True)

encode_text_dummy(df,'http.request.method')
encode_text_dummy(df,'http.referer')
encode_text_dummy(df,"http.request.version")
encode_text_dummy(df,"dns.qry.name.len")
encode_text_dummy(df,"mqtt.conack.flags")
encode_text_dummy(df,"mqtt.protoname")
encode_text_dummy(df,"mqtt.topic")
print(df.info())

'''
# Step4: map original 15 classes into 6 classes and 2 classes
print('Unique values in y_column: {}'.format(len(df['Attack_type'].unique())))
print(df['Attack_type'].unique())
df.rename(columns={'Attack_type': 'label'}, inplace=True)

# Category: DDoS, Injection, MITM, Malware, Normal, Scanning
dict_6_classes = { 'Normal': 'Normal',   
                    'DDoS_ICMP': 'DDoS', 'DDoS_HTTP': 'DDoS', 'DDoS_TCP': 'DDoS', 'DDoS_UDP': 'DDoS', 
                    'MITM': 'MITM', 
                    'XSS': 'Injection', 'SQL_injection': 'Injection', 'Uploading': 'Injection', 
                    'Backdoor': 'Malware', 'Password': 'Malware', 'Ransomware': 'Malware', 
                    'Vulnerability_scanner': 'Scanning', 'Port_Scanning': 'Scanning', 'Fingerprinting': 'Scanning'}  

dict_2_classes = {'Normal': 'Normal',   
                    'DDoS_ICMP': 'Malicious', 'DDoS_HTTP': 'Malicious', 'DDoS_TCP': 'Malicious', 'DDoS_UDP': 'Malicious', 
                    'MITM': 'Malicious', 
                    'XSS': 'Malicious', 'SQL_injection': 'Malicious', 'Uploading': 'Malicious', 
                    'Backdoor': 'Malicious', 'Password': 'Malicious', 'Ransomware': 'Malicious', 
                    'Vulnerability_scanner ': 'Malicious', 'Port_Scanning': 'Malicious', 'Fingerprinting': 'Malicious'}


df6 = df.copy()
df2 = df.copy()
df6['label'] = df6['label'].map(dict_6_classes)
df2['label'] = df2['label'].map(dict_2_classes)
print(df6['label'].value_counts())

df.to_csv(os.path.join(current_directory, '..', '..', '..', 'datasets', 'Edgedf15.csv'), index=False)
df6.to_csv(os.path.join(current_directory, '..', '..', '..', 'datasets', 'Edgedf6.csv'), index=False)
df2.to_csv(os.path.join(current_directory, '..', '..', '..', 'datasets', 'Edgedf2.csv'), index=False)
'''
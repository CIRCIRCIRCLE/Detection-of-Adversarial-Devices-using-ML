import os
import pandas as pd

def sample(df, label, sampling_ratio):
    print(label)
    df_label = df[df['label'] == label]
    sampled_df = df_label.sample(frac=sampling_ratio, random_state=42)
    #sampled_df.drop(columns=[label], inplace=True)
    df = df[df['label'] != label]
    df_sampled = pd.concat([df, sampled_df], ignore_index=True)
    return df_sampled

curdir = os.getcwd()
#curdir = os.path.dirname(__file__) 
#print(curdir)
df = pd.read_csv(os.path.join(curdir, '..', '..', '..', 'datasets', 'CIC_formatted.csv'))
filters = ['Malicious', 'Benign']
df_new = df[df['label'].isin(filters)]
cnt = df_new['label'].value_counts()
print(cnt)

df_sampled = sample(df_new, 'Malicious', 0.8)
#df_sampled = sample(df_sampled, 'DoS', 0.1)
#df_sampled = sample(df_sampled, 'Mirai', 0.3)
#df_sampled = sample(df_sampled, 'Benign', 0.7)

cnt = df_sampled['label'].value_counts()
print(cnt)
df_sampled.to_csv(os.path.join(curdir, '..', '..', '..','datasets', 'CIC_formatted.csv'))
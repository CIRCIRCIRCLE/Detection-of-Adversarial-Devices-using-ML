import pandas as pd
import os

curdir = os.getcwd()
df = pd.read_csv(os.path.join(curdir, '..', '..', '..', 'datasets', 'CIC_formatted.csv'))

# 删除第一列
df = df.iloc[:, 1:]

# 保存修改后的CSV文件
df.to_csv('CIC_formatted.csv', index=False)

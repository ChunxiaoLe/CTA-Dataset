import numpy as np
import pandas as pd

id_list = []
num_list = []
train_list = {'id':[], 'num':[], 'camera':[]}
test_list = {'id':[], 'num':[], 'camera':[]}
more_list = {'id':[], 'num':[], 'camera':[]}

df = pd.read_excel('./dataset/CTA/test.xls', sheet_name='mate30')
# 读取指定区域的两列数据
id_list = df['id'].tolist()
num_list = df['num'].tolist()

train_list['id'] = id_list
train_list['num'] = num_list
train_list['camera'] = ['mate30']

np.save('./dataset/CTA/test_mate30.npy', train_list)


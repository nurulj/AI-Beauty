# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 05:52:22 2018

@author: nurooool

"""
import h5py
import csv
import pandas as pd
import numpy as np

from collections import defaultdict
from scipy.spatial import distance
from scipy.stats import rankdata

import tensorflow as tf
tf.__version__


ff = h5py.File('features_resnet50_avgpool.hdf5', 'r')
#ff = h5py.File('features_resnet50_maxpool.hdf5', 'r')
f1 = ff['feature']
f1 = np.array(f1)
f1_t = ff['feature_labels']
train = np.array(f1_t)

tv = h5py.File('val_res50_avg.hdf5', 'r')
f2 = tv['feature']
f2 = np.array(f2)
f2_v = tv['feature_labels']
val = np.array(f2_v)

Y = distance.cdist(f1, f2, 'cosine')

res = []

#loop to rank the data by each column of Y
for i in range(len(f2)):
    rank = rankdata(Y[:,i], method='min')
    
    df = pd.DataFrame({
        'name' : train ,
        'ranks' : rank
        })
        
    result = df.sort_values(by=['ranks'])
    top7 = result[0:7]['name'].values
    res.append(top7)
    
combine = pd.DataFrame({
        'val' : val ,
        'results' : res
        })
        
combination = combine.sort_values(by=['val'])
value = combination['results'].values   
file_id = combination['val'].values
file_id = list(file_id)

with open('result-cosine.csv', "wb") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for line in value:
            writer.writerow(line)          
#%% VALIDATION
valcol= defaultdict(list)
with open('val.csv') as f:
    reader = csv.DictReader(f) # read rows into a dictionary format
    for row in reader: # read a row as {column1: value1, column2: value2,...}
        for (k,v) in row.items(): # go over each column name and value 
            valcol[k].append(v)

#%%
rescol = defaultdict(list)
with open('result-cosine.csv') as f:
    reader = csv.DictReader(f) # read rows into a dictionary format
    for row in reader: # read a row as {column1: value1, column2: value2,...}
        for (k,v) in row.items(): # go over each column name and value 
            rescol[k].append(v)
            
#%% Loop to check
comb_val = np.array([valcol['Train_ID1'], valcol['Train_ID2'], valcol['Train_ID3']]).T
comb_res = np.array([rescol['Top1'], rescol['Top2'], rescol['Top3'], rescol['Top4'], 
                     rescol['Top5'], rescol['Top6'], rescol['Top7']]).T

def common_member(a, b):
    a_set = set(a)
    b_set = set(b)
    if (a_set & b_set):
        return True
    else:
        return False

yes = 0
no = 0

for i in range(len(comb_val)):
    check = common_member(comb_val[i], comb_res[i])
    if check == True:
        yes = yes + 1
    else:
        no = no + 1
        

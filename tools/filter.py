#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 22:23:15 2019

@author: apple
"""

import tqdm
import pandas as pd
import os
import matplotlib.pyplot as plt

def main():
    data_dir = './data'
    df_train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    num = df_train.shape[0]
    print('total ', num)
    
    landmark_id_list = {}
    for i in tqdm.tqdm(range(num)):
        landmark_id = df_train.get_value(i, 'landmark_id')
        if landmark_id not in landmark_id_list:
            landmark_id_list[landmark_id] = 1
        else:
            landmark_id_list[landmark_id] += 1
    
    total = len(landmark_id_list)   
    print('landmark_len ', total)
  
    filter_list = []
    keys = landmark_id_list.keys()
    
    for i, key in enumerate (keys):
        value = landmark_id_list[key]
        if value >= 5:
           filter_list.append(key)
           
    print('landmark_len after ', len(filter_list))
    
    train_data = []
    for i in tqdm.tqdm(range(num)):
        img_id = df_train.get_value(i, 'id')
        landmark_id = df_train.get_value(i, 'landmark_id')
        if landmark_id in filter_list:
            train_data.append((img_id, landmark_id))
            
    train_pd = pd.DataFrame.from_records(train_data, columns=['id', 'landmark_id'])
    output_filename = os.path.join(data_dir, 'filter_data_train.csv')
    train_pd.to_csv(output_filename, index=False)
    
if __name__ == '__main__':
  main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 14:59:26 2019

@author: zl
"""
import tqdm
import pandas as pd
import os

def main():
    data_dir = './data'
    df_train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    num = df_train.shape[0]
    print('total ', num)
    
    landmark_id_list = []
    for i in tqdm.tqdm(range(num)):
        landmark_id = df_train.get_value(i, 'landmark_id')
        if landmark_id not in landmark_id_list:
            landmark_id_list.append(landmark_id)
            
    print('landmark_len ', len(landmark_id_list))
if __name__ == '__main__':
  main()
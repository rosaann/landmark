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
    landmark_id_list = sorted(landmark_id_list.items())
    plt.bar(range(len(landmark_id_list)), landmark_id_list)
    plt.savefig("assign.jpg")
    print ('land ', landmark_id_list)   
        
if __name__ == '__main__':
  main()
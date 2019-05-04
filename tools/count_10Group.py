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
import random


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
    keys = list(landmark_id_list.keys())
    random.shuffle (keys )

    group_item_num = 1000
    key_group_list = []
    for i, key in enumerate( keys):
        if i % group_item_num == 0 and (i / 1000) < 203:
            key_group = []
        key_group.append(key)
        if i % group_item_num == (group_item_num - 1) and (i / 1000) < 203:
            key_group_list.append(key_group)
            print('group ', i / group_item_num, ' len ', len(key_group))
            
         
        
  #  fileObject = open('assign.txt', 'w')
  #  for t in landmark_id_list.items():
  #      fileObject.write(str(t))
  #      fileObject.write('\n')
  #  fileObject.close()
  #  plt.bar(keys, values)
  #  plt.savefig("assign.jpg")
  #  print ('land ', landmark_id_list)   
        
if __name__ == '__main__':
  main()
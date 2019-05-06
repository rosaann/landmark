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
import ast
data_dir = './data'
group_item_num = 1000
def genkeyGroups():
    
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

    
    key_group_list = []
    group_num = len(keys) / group_item_num
    key_group = []
    for i, key in enumerate( keys):
        if i % group_item_num == 0 and (i / group_item_num) < group_num:
            key_group = []
        key_group.append(key)
        if i % group_item_num == (group_item_num - 1) and (i / group_item_num) !=  group_num:
            key_group_list.append(key_group)
           # print('group ', i / group_item_num, ' len ', len(key_group))
    key_group_list[-1].extend(key_group)
  #  print('group last', ' len ', len(key_group_list[-1]))
    fileObject = open(os.path.join(data_dir, 'key_groups.txt'), 'w')
  #  for ip in key_group_list:
    fileObject.write(str(key_group_list))
    fileObject.write('\n')
    fileObject.close()
    
def write_train_val_to_csv(datalist, groupIdx):
    num = len(datalist)
    index_list = list(range(num))
    random.shuffle(index_list)
    train_num = int(num * 0.7)
    val_num = int(num * 0.3)
  #  test_num = num - train_num - val_num
    
    train_data = []
    for i in tqdm.tqdm(range(train_num)):
        img_id = datalist[i][0]
        landmark_id = datalist[i][1]
        train_data.append((img_id, landmark_id))
    
    train_pd = pd.DataFrame.from_records(train_data, columns=['id', 'landmark_id'])
    file_name = 'data_train_group_' + str(groupIdx) + '.csv'
    output_filename = os.path.join(data_dir, file_name)
    train_pd.to_csv(output_filename, index=False)
    
    val_data = []
    for i in tqdm.tqdm(range(val_num)):
        img_id = datalist[i][0]
        landmark_id = datalist[i][1]
        val_data.append((img_id, landmark_id))
    
    val_pd = pd.DataFrame.from_records(val_data, columns=['id', 'landmark_id'])
    val_file_name = 'data_tval_group_' + str(groupIdx) + '.csv'
    output_filename = os.path.join(data_dir, val_file_name)
    val_pd.to_csv(output_filename, index=False)
def gen_data_cvs_for_group():
    key_group_list = []
    with open(os.path.join(data_dir, 'key_groups.txt'), 'r') as f: 
         key_group_list = ast.literal_eval(f.read())
         
    df_train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    num = df_train.shape[0]
    for gi, key_group in enumerate( tqdm.tqdm(key_group_list)):
        key_in_data = []
        key_not_data = []
        for i in tqdm.tqdm(range(num)):
            img_id = df_train.get_value(i, 'id')
            landmark_id = df_train.get_value(i, 'landmark_id')
            if landmark_id in key_group:
                key_in_data.append((img_id, landmark_id))
            else:
                key_not_data.append((img_id, landmark_id))
                
        random.shuffle(key_not_data)
        key_in_data.extend(key_not_data[:int(len(key_in_data) / 2)])
        random.shuffle(key_in_data)
        write_train_val_to_csv(key_in_data, gi)
def main():
   # genkeyGroups()
    gen_data_cvs_for_group()
    
        
        
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
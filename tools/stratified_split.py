from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse

import random

import tqdm
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', dest='data_dir',
                        help='the directory of the data',
                        default='', type=str)
    return parser.parse_args()


def main():
  #  args = parse_args()
    data_dir = './data'
    raw_images_dir = os.path.join(data_dir, 'train_images')
    df_train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    num = df_train.shape[0]
    print('total ', num)
    index_list = list(range(num))
    random.shuffle(index_list)
    train_num = int(num * 0.7)
    val_num = int(num * 0.3)
  #  test_num = num - train_num - val_num
    
    train_data = []
    for i in tqdm.tqdm(range(train_num)):
        img_id = df_train.get_value(i, 'id')
        landmark_id = df_train.get_value(i, 'landmark_id')
        train_data.append((img_id, landmark_id))
    
    train_pd = pd.DataFrame.from_records(train_data, columns=['id', 'landmark_id'])
    output_filename = os.path.join(data_dir, 'data_train.csv')
    train_pd.to_csv(output_filename, index=False)
    
    val_data = []
    for i in tqdm.tqdm(range(val_num)):
        img_id = df_train.get_value(i, 'id')
        landmark_id = df_train.get_value(i, 'landmark_id')
        val_data.append((img_id, landmark_id))
    
    val_pd = pd.DataFrame.from_records(val_data, columns=['id', 'landmark_id'])
    output_filename = os.path.join(data_dir, 'data_val.csv')
    val_pd.to_csv(output_filename, index=False)

   # test_data = []
   # for i in tqdm.tqdm(range(test_num)):
   #     img_id = df_train.get_value(i, 'ImageId')
   #     encoder_p = df_train.get_value(i, 'EncodedPixels')
  #      test_data.append((img_id, encoder_p))
    
  #  test_pd = pd.DataFrame.from_records(test_data, columns=['ImageId', 'EncodedPixels'])
  #  output_filename = os.path.join(raw_images_dir, 'data_test.csv')
  #  test_pd.to_csv(output_filename, index=False)
    
if __name__ == '__main__':
  main()


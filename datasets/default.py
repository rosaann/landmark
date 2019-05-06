from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import pandas as pd
import scipy.misc as misc

from torch.utils.data.dataset import Dataset
import tqdm
import ast

class DefaultDataset(Dataset):
    def __init__(self,
                 group_idx,
                 split,
                 transform=None,
                 idx_fold=0,
                 num_fold=5,
                 #split_prefix='split.stratified',
                 **_):
        self.split = split
        self.group_idx = group_idx
        self.idx_fold = idx_fold
        self.num_fold = num_fold
        self.transform = transform
        self.dataset_dir = './data/group_csv/'
        self.images_dir = os.path.join('./data/', 'train_images')
        csv_path = 'data_{}_group_{}.csv'.format(self.split, self.group_idx)
        self.csv_path = os.path.join(self.dataset_dir, csv_path)
        
        self.df_labels = self.load_labels()
        self.load_key_idx()

        self.size = len(self.key2idx)
        
    def load_key_idx(self):
        key_group_list = []
        with open(os.path.join('./data/', 'key_groups.txt'), 'r') as f: 
            key_group_list = ast.literal_eval(f.read())
        self.key_list =  key_group_list[self.group_idx]
        key_idx_list = {}
        for i, key in enumerate( self.key_list):
            key_idx_list[key] = i
        
        df_train = pd.read_csv(self.csv_path)
        num = df_train.shape[0]
        print('total ', num)
        self.key2idx = []
        key_idx_list = {}
        for i in tqdm.tqdm(range(num)):
            landmark_id = df_train.get_value(i, 'landmark_id')
            if landmark_id not in self.key_list:
                 self.key2idx.append( len(key_idx_list.items()))
            else:
                self.key2idx.append(key_idx_list[landmark_id])
                
        
            
    def load_labels(self):
        
      #  print('labels_path ', labels_path)
        df_labels = pd.read_csv(self.csv_path)
     #   print('df_labels ', df_labels)
     #   df_labels = df_labels.reset_index()

        def to_filepath(v):
            dir1 = v[0:1]
            dir2 = v[1:2]
            dir3 = v[2:3]
            return os.path.join(self.images_dir,dir1,dir2,dir3, v + '.jpg')
            

        df_labels['filepath'] = df_labels['id'].transform(to_filepath)
        #print(df_labels)
        return df_labels

    
                

    def __getitem__(self, index):

        filename = self.df_labels[index]
        image = misc.imread(filename)

        if self.transform is not None:
            image = self.transform(image)
       # print('filename ', filename, ' key ', example[0])
        return {'image': image,
                'key': self.key2idx[index]}

    def __len__(self):
        return self.size


def test():
    dataset = DefaultDataset('data', 'train', None)
    print(len(dataset))
    example = dataset[0]
    example = dataset[1]

    dataset = DefaultDataset('data', 'val', None)
    print(len(dataset))

if __name__ == '__main__':
    test()

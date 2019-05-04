from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import pandas as pd
import scipy.misc as misc

from torch.utils.data.dataset import Dataset
import tqdm

class DefaultDataset(Dataset):
    def __init__(self,
                 dataset_dir,
                 split,
                 transform=None,
                 idx_fold=0,
                 num_fold=5,
                 #split_prefix='split.stratified',
                 **_):
        self.split = split
        self.idx_fold = idx_fold
        self.num_fold = num_fold
        self.transform = transform
        self.dataset_dir = dataset_dir
        self.images_dir = os.path.join(dataset_dir, 'train_images')

        self.df_labels = self.load_labels()
        self.load_key_idx()
        self.examples = self.load_examples()
        self.size = len(self.examples)
        
    def load_key_idx(self):
        whole_path = os.path.join(self.dataset_dir, 'filter_data_train.csv')
        df_train = pd.read_csv(whole_path)
        num = df_train.shape[0]
        print('total ', num)
        key_idx_list = {}
        for i in tqdm.tqdm(range(num)):
            landmark_id = df_train.get_value(i, 'landmark_id')
            if landmark_id not in key_idx_list:
                key_idx_list[landmark_id] = len(key_idx_list.items())
                
        self.key2idx = key_idx_list
            
    def load_labels(self):
        labels_path = 'data_{}.csv'.format(self.split)
        labels_path = os.path.join(self.dataset_dir, labels_path)
      #  print('labels_path ', labels_path)
        df_labels = pd.read_csv(labels_path)
     #   print('df_labels ', df_labels)
        df_labels = df_labels.reset_index()

        def to_filepath(v):
            dir1 = v[0:1]
            dir2 = v[1:2]
            dir3 = v[2:3]
            return os.path.join(self.images_dir,dir1,dir2,dir3, v + '.jpg')
            

        df_labels['filepath'] = df_labels['id'].transform(to_filepath)
        #print(df_labels)
        return df_labels

    def load_examples(self):
        return [(self.key2idx[row['landmark_id']], row['filepath'])
                for _, row in self.df_labels.iterrows()]

    def __getitem__(self, index):
        example = self.examples[index]

        filename = example[1]
        image = misc.imread(filename)

        if self.transform is not None:
            image = self.transform(image)
       # print('filename ', filename, ' key ', example[0])
        return {'image': image,
                'key': example[0]}

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

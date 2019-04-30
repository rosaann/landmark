from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import pandas as pd
import scipy.misc as misc

from torch.utils.data.dataset import Dataset


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
        self.examples = self.load_examples()
        self.size = len(self.examples)

    def load_labels(self):
        labels_path = 'data_{}.csv'.format(self.split)
        labels_path = os.path.join(self.dataset_dir, labels_path)
      #  print('labels_path ', labels_path)
        df_labels = pd.read_csv(labels_path)
        print('df_labels ', df_labels)
        df_labels = df_labels.reset_index()

        def to_filepath(v):          
            return os.path.join(self.images_dir, v + '.jpg')
            

        df_labels['filepath'] = df_labels['id'].transform(to_filepath)
        return df_labels

    def load_examples(self):
        return [(row['landmark_id'], row['filepath'])
                for _, row in self.df_labels.iterrows()]

    def __getitem__(self, index):
        example = self.examples[index]

        filename = example[1]
        image = misc.imread(filename)

        if self.transform is not None:
            image = self.transform(image)

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

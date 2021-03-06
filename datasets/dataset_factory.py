from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import numpy as np
import torch.utils.data
import torch.utils.data.sampler
from torch.utils.data import DataLoader

from .default import DefaultDataset
from .small import SmallDataset
from .test_dataset import TestDataset


def get_dataset(config,group_idx, split, transform=None, last_epoch=-1):
    f = globals().get(config.name)

    return f(group_idx,
             split=split,
             transform=transform,
             **config.params)


def get_dataloader(config, group_idx, split, transform=None, **_):
    dataset = get_dataset(config.data, group_idx, split, transform)

    is_train = 'train' == split
    batch_size = config.train.batch_size if is_train else config.eval.batch_size

    print('split ', split, ' batch_size ', batch_size)
    dataloader = DataLoader(dataset,
                            shuffle=is_train,
                            batch_size=batch_size,
                          #  drop_last=is_train,
                            
                            num_workers=config.transform.num_preprocessor,
                            pin_memory=False)
    return dataloader

def get_test_loader(config, img_id_list, transform=None):
        dataset = TestDataset(img_id_list, transform)
        batch_size = config.test.batch_size
        dataloader = DataLoader(dataset,
                            shuffle=False,
                            batch_size=batch_size,                            
                            num_workers=8,
                            pin_memory=False)
        
        return dataloader

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 11:14:16 2019

@author: zl
"""

from download import download_file
def down():
    #file = 'https://s3.amazonaws.com/google-landmark/metadata/train.csv'
    file = 'https://s3.amazonaws.com/google-landmark/metadata/train_attribution.csv'
    download_file(file, './train_attribution.csv')
    

    
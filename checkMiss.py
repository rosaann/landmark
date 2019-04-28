#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 09:17:37 2019

@author: zl
"""

import os
basePath = './images/'
paths = os.listdir(basePath)
path_idx_list = []
for path in paths:
    path = path.replace('images_00', '')
    path = path.replace('images_0', '')
    path = path.replace('images_', '')
    path = path.replace('.tar', '')
    path_idx_list.append(int(path))
    
miss = []    
for i in range(500):
    if i not in path_idx_list:
        miss.append(i)
        
print(miss)
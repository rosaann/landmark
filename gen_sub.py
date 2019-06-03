#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 18:48:38 2019

@author: zl
"""
import pandas as pd

def main():
    df_test_img_land = pd.read_csv(os.path.join('', 'test_img_land.csv.csv'))
    num = df_test_img_land.shape[0]
    
    results = {}
    for i in tqdm.tqdm(range(num)):
        img_id = df_test_img_land.get_value(i, 'image')
        landmark_id = df_test_img_land.get_value(i, 'img_id')
        if results.has_key(landmark_id):
            results[landmark_id].append(img_id)
        else:
            results[landmark_id] = [img_id]

if __name__ == '__main__':
    main()
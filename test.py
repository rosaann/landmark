#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 09:20:02 2019

@author: zl
"""
import csv, os, ast, tqdm
test_data_file = 'data/test_csv/test.csv'
test_img_download_fail_list = []
test_img_list = []
def ParseData(data_file):
  csvfile = open(data_file, 'r')
  csvreader = csv.reader(csvfile)
  key_url_list = [line[:2] for line in csvreader]
  return key_url_list[1:]

def getDownFailedImgs():
    down_failed_file =  'down_failed.txt'
    down_failed_list = {}
    if os.path.exists(down_failed_file):
        with open(down_failed_file, 'r') as f: 
            down_failed_list = ast.literal_eval(f.read())
            return down_failed_list
        
def getTestImgList():
    key_url_list = ParseData(test_data_file)
    test_img_download_fail_list = getDownFailedImgs()
    for img_key, url in tqdm.tqdm(key_url_list):
        if len(url) < 10:
            continue
        if img_key not in test_img_download_fail_list:
            test_img_list.append(img_key)
        

def main():
    #获取testimg列表
    test_img_list = getTestImgList()

if __name__ == '__main__':
    main()
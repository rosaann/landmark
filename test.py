#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 09:20:02 2019

@author: zl
"""
import csv, os, ast, tqdm
from datasets import get_test_loader
import torch
from models import get_model
import utils
import utils.config
import utils.checkpoint
from optimizers import get_optimizer
import argparse

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

def inference(model, images):
    logits = model(images)
  #  print('logits ', logits)
    if isinstance(logits, tuple):
        logits, aux_logits = logits
    else:
        aux_logits = None
    probabilities = F.sigmoid(logits)
  #  print('probabilities ', probabilities)
    return logits, aux_logits, probabilities
     
def test_one_model(dataloader, model, group_key_list, result_set):
    tbar = tqdm.tqdm(enumerate(dataloader))
    for i, data in tbar:
        images = data['image']
        img_ids = data['img_id']
        if torch.cuda.is_available():
            images = images.cuda()
            img_ids = img_ids.cuda()
            
        logits, aux_logits, probabilities = inference(model, images)
        for img_i, img_id in enumerate(img_ids):
            real_landmark_id = group_key_list[img_i]
            result_set[img_id][real_landmark_id] = probabilities[img_i]
            
    return result_set

def parse_args():
    parser = argparse.ArgumentParser(description='HPA')
    parser.add_argument('--config', dest='config_file',
                        help='configuration filename',
                        default=None, type=str)
    return parser.parse_args()

def main():
    args = parse_args()
    if args.config_file is None:
      raise Exception('no configuration file')

    config = utils.config.load(args.config_file)
    #获取testimg列表
    test_img_list = getTestImgList()
    test_data_set = get_test_loader(config, test_img_list)
    result = {}
    for img_id in test_img_list:
        #初始化添加
        result[img_id] = {}
    
    with open(os.path.join(data_dir, 'key_groups.txt'), 'r') as f: 
         key_group_list = ast.literal_eval(f.read())
    
    best_model_idx_dic = {}     
    for gi in range(204):
        best_model_idx_dic[gi] = 13
        
    for gi, key_group in enumerate( tqdm.tqdm(key_group_list)):
        model = get_model(config, gi)
        if torch.cuda.is_available():
            model = model.cuda()
        optimizer = get_optimizer(config, model.parameters())
        checkpoint = utils.checkpoint.get_model_saved(config, gi, best_model_idx_dic[gi])
        best_epoch, step = utils.checkpoint.load_checkpoint(model, optimizer, checkpoint)
        result = test_one_model(test_data_set, model, key_group, result)
        #

if __name__ == '__main__':
    main()
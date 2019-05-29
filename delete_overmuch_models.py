#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 18:42:26 2019

@author: zl
"""
import os

def del_models_of_group(gi):
    gi_dir = os.path.join('./results/attention_inceptionv3/writer', str(gi))
    if  not os.path.exists(gi_dir):
        return
    checkpoint_dir = os.path.join(gi_dir, 'checkpoint')
    for checkpoint in os.listdir(checkpoint_dir):
        if checkpoint.startswith('epoch_') and checkpoint.endswith('.pth'):
            print('checkpoint ', checkpoint)
            gstr = checkpoint.replace('epoch_', '')
            gstr = checkpoint.replace('.pth', '')
            n = int(gstr)
            if n < 13:
                f = os.path.join(checkpoint_dir, checkpoint)
                print('f', f)
             #   os.remove(f)
            

        

def main():
    gi_list = range(206)
    for gi in gi_list:
        del_models_of_group(gi)

if __name__ == '__main__':
    main()
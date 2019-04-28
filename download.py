#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 16:26:38 2019

@author: zl
"""

import requests
import urllib 
import sys
import time
import os
import socket
socket.setdefaulttimeout(30)
base = 'https://s3.amazonaws.com/google-landmark/train/'
url=''
img = ''
def callbackfunc(blocknum, blocksize, totalsize):
    '''回调函数
    @blocknum: 已经下载的数据块
    @blocksize: 数据块的大小
    @totalsize: 远程文件的大小
    '''
    global url
    percent = 100.0 * blocknum * blocksize / totalsize
    if percent > 100:
        percent = 100
    downsize=blocknum * blocksize
    if downsize >= totalsize:
    	downsize=totalsize
    s ="%.2f%%"%(percent)+"====>"+"%.2f"%(downsize/1024/1024)+"M/"+"%.2f"%(totalsize/1024/1024)+"M \r"
    sys.stdout.write(s)
    sys.stdout.flush()
    if percent == 100:
        print('')
        #input('输入任意键继续...')
def download(url, file,callbackfunc):
    urllib.request.urlretrieve(url, file, callbackfunc)

#miss = [66, 200, 203, 204, 205, 207, 269, 321, 322, 337, 341, 355, 356, 360, 362, 416, 485, 499]
miss = [0, 203, 204, 205, 207, 321]
#for i in range(500)[69:500]:
for i in miss:
    if i < 10:
        img = 'images_00' + str(i) + '.tar'
        url = base + img       
    else :
        if i < 100:
            img = 'images_0' + str(i) + '.tar'
            url = base + img    
        else:
            img = 'images_' + str(i) + '.tar'
            url = base + img    
    
    print('downloading ', url)
    for ti in range(5):
        try:
            download(url, 'images/' + img, callbackfunc)
            break
        except Exception as e:
            file = 'images/' + img
            if os.path.exists(file):              
                os.remove(file)
            time.sleep(60)
    time.sleep(60)
    
    #https://s3.amazonaws.com/google-landmark/train/images_000.tar
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 09:51:16 2019

@author: zl
"""

#!/usr/bin/python

# Note to Kagglers: This script will not run directly in Kaggle kernels. You
# need to download it and run it on your local machine.

# Downloads images from the Google Landmarks dataset using multiple threads.
# Images that already exist will not be downloaded again, so the script can
# resume a partially completed download. All images will be saved in the JPG
# format with 90% compression quality.

import sys, os, multiprocessing, csv
from PIL import Image
from io import BytesIO
from urllib.request import urlopen


def ParseData(data_file):
  csvfile = open(data_file, 'r')
  csvreader = csv.reader(csvfile)
  key_url_list = [line[:2] for line in csvreader]
  return key_url_list[1:]  # Chop off header

out_dir = './data/test_images'
def DownloadImage(key_url):
  
  (key, url) = key_url
  print('key ', key)
  filename = os.path.join(out_dir, '%s.jpg' % key)

  if os.path.exists(filename):
    print('Image %s already exists. Skipping download.' % filename)
    return

  try:
    response = urlopen(url)
    image_data = response.read()
  except:
    print('Warning: Could not download image %s from %s' % (key, url))
    return

  try:
    pil_image = Image.open(BytesIO(image_data))
  except:
    print('Warning: Failed to parse image %s' % key)
    return

def Run():
 # if len(sys.argv) != 3:
 #   print('Syntax: %s <data_file.csv> <output_dir/>' % sys.argv[0])
 #   sys.exit(0)
 # (data_file, out_dir) = sys.argv[1:]
  data_file = './test_csv/test.csv'
  #out_dir = './test_images'

  if not os.path.exists(out_dir):
    os.mkdir(out_dir)

  key_url_list = ParseData(data_file)
  pool = multiprocessing.Pool(processes=10)
  pool.map(DownloadImage, key_url_list)


if __name__ == '__main__':
  Run()
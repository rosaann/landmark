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
import time
import tqdm
import urllib.request
import ast
import socket
socket.setdefaulttimeout(30)
def ParseData(data_file):
  csvfile = open(data_file, 'r')
  csvreader = csv.reader(csvfile)
  key_url_list = [line[:2] for line in csvreader]
  return key_url_list[1:]  # Chop off header

out_dir = 'data/test_images'
 
down_failed_file =  'down_failed.txt'
down_failed_list = {}
if os.path.exists(down_failed_file):
    with open(down_failed_file, 'r') as f: 
         down_failed_list = ast.literal_eval(f.read())


def note_down_failed(key, url):
    down_failed_list[key] = url
    fileObject = open(down_failed_file, 'w')
  #  for ip in key_group_list:
    fileObject.write(str(down_failed_list))
    fileObject.write('\n')
    fileObject.close()         
def DownloadImage(key_url):
  
  (key, url) = key_url
  filename = os.path.join(out_dir, '%s.jpg' % key)

  if os.path.exists(filename):
    print('Image %s already exists. Skipping download.' % filename)
    return
  if key in down_failed_list.keys():
     print('Image %s already in download failed list. Skipping download.' % filename)
     return
  if len(url) < 10:
      print('Image %s url none. Skipping download.' % filename)
      return
  
  total = 3
  for ti in range(3):
    try:
      print('link Image %s .' % key)
      #response = urlopen(url)
      #image_data = response.read()
      image_data = urllib.request.urlopen(url).read()
      break
    except:
      print('Warning: Could not download image %s from %s' % (key, url))
      if ti == (total - 1):
          note_down_failed(key, url)
          print('Warning: download failed, note %s from %s' % (key, url))
      else:
          time.sleep(10)
    
  try:
    pil_image = Image.open(BytesIO(image_data))
  except:
    print('Warning: Failed to parse image %s' % key)
    


  try:
    pil_image.save(filename, format='JPEG', quality=90)
    print('save Image %s .----' % key)
  except:
    print('Warning: Failed to save image %s' % filename)
    
  time.sleep(1)
  

def Run():
#  if len(sys.argv) != 3:
#    print('Syntax: %s <data_file.csv> <output_dir/>' % sys.argv[0])
#    sys.exit(0)
 # (data_file, out_dir) = sys.argv[1:]
  data_file = 'data/test_csv/test.csv'
  if not os.path.exists(out_dir):
    os.mkdir(out_dir)

  key_url_list = ParseData(data_file)
  for key_url in tqdm.tqdm(key_url_list[: :-1]):
      DownloadImage(key_url)
  #pool = multiprocessing.Pool(processes=1)
  #pool.map(DownloadImage, key_url_list)


if __name__ == '__main__':
  Run()
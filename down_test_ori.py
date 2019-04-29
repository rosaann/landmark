#!/usr/bin/python

# Note to Kagglers: This script will not run directly in Kaggle kernels. You
# need to download it and run it on your local machine.

# Downloads images from the Google Landmarks dataset using multiple threads.
# Images that already exist will not be downloaded again, so the script can
# resume a partially completed download. All images will be saved in the JPG
# format with 90% compression quality.

import sys, os, multiprocessing, urllib2, csv
from PIL import Image
from io import BytesIO
from urllib.request import urlopen
import time

def ParseData(data_file):
  csvfile = open(data_file, 'r')
  csvreader = csv.reader(csvfile)
  key_url_list = [line[:2] for line in csvreader]
  return key_url_list[1:]  # Chop off header

out_dir = 'data/test_images'
def DownloadImage(key_url):
  
  (key, url) = key_url
  filename = os.path.join(out_dir, '%s.jpg' % key)

  if os.path.exists(filename):
    print('Image %s already exists. Skipping download.' % filename)
    return
  for ti in range(5):
    try:
      print('Image %s .' % key)
      response = urlopen(url)
      image_data = response.read()
      break
    except:
      print('Warning: Could not download image %s from %s' % (key, url))
      time.sleep(10)

  try:
    pil_image = Image.open(BytesIO(image_data))
  except:
    print('Warning: Failed to parse image %s' % key)
    return


  try:
    pil_image.save(filename, format='JPEG', quality=90)
  except:
    print('Warning: Failed to save image %s' % filename)
    return


def Run():
  if len(sys.argv) != 3:
    print('Syntax: %s <data_file.csv> <output_dir/>' % sys.argv[0])
    sys.exit(0)
 # (data_file, out_dir) = sys.argv[1:]
  data_file = 'data/test_csv/test.csv'
  if not os.path.exists(out_dir):
    os.mkdir(out_dir)

  key_url_list = ParseData(data_file)
  pool = multiprocessing.Pool(processes=10)
  pool.map(DownloadImage, key_url_list)


if __name__ == '__main__':
  Run()
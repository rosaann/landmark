#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 09:20:02 2019

@author: zl
"""
import urllib.request
url='https://lh3.googleusercontent.com/-q8B91vDIQZY/WM-q-ZAfhDI/AAAAAAAAGYw/wr1Cn1kzSCkC5uX_zbkGyn7pYzCzng6dgCOcB/s1600/'
contents = urllib.request.urlopen(url).read()
print(contents)
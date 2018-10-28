#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 18:55:56 2018

@author: ashwin
"""

import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

work_dir='/home/ashwin/Desktop/Sign-Language-Digits-Dataset-master/Dataset/'

lables=os.listdir(work_dir)
X=[]
y=[]
img_size=64

for l in lables:
    path=work_dir+l+'/'
    img_list=os.listdir(path)
    for img in img_list:
        x=cv.imread(path+img,0)
        x = cv.resize(x, (img_size, img_size))
        X.append(x)
        y.append(l)
        
y=np.asarray(y)
X=np.asarray(X)
np.save('X_data.npy',X)
np.save('y_data.npy',y)

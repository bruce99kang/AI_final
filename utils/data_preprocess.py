#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 16:25:50 2020

@author: bruce
"""

import numpy as np
import tensorflow as tf
import pandas as pd
import subprocess
import glob
from tqdm import tqdm

lis_attr = open('/home/bruce/Downloads/CelebA/CelebA-20201202T065512Z-015/CelebA/Anno/list_attr_celeba.txt')
txt = lis_attr.readlines()

tmp = txt[2].split(' ')

with_hair = []
count0,count1,count2,count3 = 0,0,0,0
for i in range(2,len(txt)):
    labels = txt[i].split(' ')
    if labels[9] == '1':
        with_hair.append(i)
        count0+=1
    elif labels[10] == '1':
        with_hair.append(i)
        count1+=1
    elif labels[12] == '1':
        with_hair.append(i)
        count2+=1
    elif labels[18] == '1':
        with_hair.append(i)
        count3+=1

original_path = '/home/bruce/Downloads/CelebA/CelebA-20201202T065512Z-015/CelebA/Img/img_align_celeba/'
to_path = '/home/bruce/AI/data/train/'
for i in tqdm(range(len(with_hair))):
    labels = txt[with_hair[i]].split(' ')
    if labels[9] == '1':
        builder = 'cp '+original_path+labels[0]+' '+to_path+'black/'
        # print(builder)
    elif labels[10] == '1':
        builder = 'cp '+original_path+labels[0]+' '+to_path+'blonde/'
        # print(builder)
    elif labels[12] == '1':
        builder = 'cp '+original_path+labels[0]+' '+to_path+'brown/'
        # print(builder)
    elif labels[18] == '1':
        builder = 'cp '+original_path+labels[0]+' '+to_path+'grey/'
        # print(builder)
    subprocess.call(builder,shell=True)

from sklearn.model_selection import train_test_split

names = ['blonde','black','brown','grey']
for name in names:
    train_lis = [f for f in glob.glob('/home/bruce/AI/data/train/'+name+'/*.jpg')]
    train_lis.sort()
    train_data, test_data = train_test_split(train_lis,test_size=0.2, random_state=1)
    train_data, val_data = train_test_split(train_data, test_size=0.125, random_state=1)
    for file1 in tqdm(val_data):
        builder = 'mv '+ file1 +' /home/bruce/AI/data/val/'+name
        # print(builder)
        subprocess.call(builder,shell=True)
    for file2 in tqdm(test_data):
        builder = 'mv '+ file2 +' /home/bruce/AI/data/test/'+name
        subprocess.call(builder,shell=True)
    print(name,'finished----------------------------------------')

# Data Balancing
black = [file for file in glob.glob('/home/bruce/AI/data/train/black/*')]
blonde = [file for file in glob.glob('/home/bruce/AI/data/train/blonde/*')]
brown = [file for file in glob.glob('/home/bruce/AI/data/train/brown/*')]
black.sort()
blonde.sort()

for i in blonde[len(brown):]:
    builder = 'mv '+str(i)+ ' /home/bruce/AI/data/tmp/blonde/'
    print(builder)
    subprocess.call(builder,shell=True)

for i in black[len(brown):]:
    builder = 'mv '+str(i)+ ' /home/bruce/AI/data/tmp/black/'
    print(builder)
    subprocess.call(builder,shell=True)
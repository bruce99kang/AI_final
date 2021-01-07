#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 21:20:44 2020

@author: bruce
"""

import numpy as np
import pandas as pd

def to_csv(lis_attr,lines,attrs):
    file_name = []
    for i in range(2, len(lines)):
        file_name.append(lines[i].split()[0])

    file_label = []
    to_use = get_index(lis_attr,attrs)

    for i in range(2, len(lines)):
        tmp = lines[i].split()[1:]
        test = np.zeros([len(lines)])
        for k in range(len(tmp)):
            if tmp[k] == '1':
                test[k] = 1
            new = np.zeros([len(to_use)])
            for j in range(len(to_use)):
                new[j] = test[to_use[j]]
        file_label.append(new)

    return file_name,file_label

def get_index(lis_attr,attrs):
    tmp=[]
    for i in range(len(lis_attr)):
        tmp.append(attrs.index(lis_attr[i]))
    tmp.sort()
    return tmp
lis_attr = open('/home/bruce/Downloads/CelebA/CelebA-20201202T065512Z-015/CelebA/Anno/list_attr_celeba.txt')
txt = lis_attr.readlines()



columns = txt[1].split(' ')[:-1]
choose = ['Bangs','Big_Nose','Bushy_Eyebrows','Chubby','Eyeglasses','Mustache']
file_name, file_label = to_csv(choose,txt,columns)
file_label = np.array(file_label)
df = pd.DataFrame({'images':file_name,'Bangs':file_label[:,0],
                   'Big_Nose':file_label[:,1],
                   'Bushy_Eyebrows':file_label[:,2],
                   'Chubby':file_label[:,3],
                   'Eyeglasses':file_label[:,4],
                   'Mustache':file_label[:,5]})

df.to_csv('total_dataframe.csv')
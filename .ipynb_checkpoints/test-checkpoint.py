#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 11:58:33 2020

@author: bruce
"""

import tensorflow as tf
import math
import numpy as np
import pandas as pd
import time
from tensorflow.keras import preprocessing
from model.model import *
import cv2

model = tf.keras.models.load_model('./work_dir/my_model_6class_baseline.h5')
data_test = pd.read_csv('./data_dataframe/total_dataframe.csv',header=0)
to_train = math.floor(0.8*len(data_test))
to_val = math.floor(0.9*len(data_test))
choose = ['Bangs','Big_Nose','Bushy_Eyebrows','Chubby','Eyeglasses','Mustache']
test_datagen = preprocessing.image.ImageDataGenerator(rescale=1./255)


testing_generator = test_datagen.flow_from_dataframe(
    dataframe=data_test.iloc[to_val:],
    directory='/home/bruce/Downloads/CelebA/CelebA-20201202T065512Z-015/CelebA/Img/img_align_celeba/',
    x_col='images',
    y_col=choose,
    target_size=(300,300),
    class_mode='other',
    batch_size=64)

start_time = time.time()

testing_history = model.evaluate_generator(testing_generator,
                              steps=len(testing_generator),
                              use_multiprocessing=True,
                              max_queue_size=128,
                              workers=8)

print('Computational Cost: ', time.time()-start_time)
print('Testing Loss: ', testing_history[0])
print('Testing Accuracy: ',testing_history[1])

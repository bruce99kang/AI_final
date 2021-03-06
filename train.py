#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 12:04:52 2020

@author: bruce
"""

import time
import pandas as pd
from tensorflow.keras import preprocessing
import math
from model.model import *


parser = argparse.ArgumentParser()
parser.add_argument('--result_dir', help='model path to save', required=True, type=str,dest='path')

# Data import
data_test = pd.read_csv('./data_dataframe/total_dataframe.csv',header=0)

train_datagen = preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
    )

test_datagen = preprocessing.image.ImageDataGenerator(rescale=1./255)

to_train = math.floor(0.8*len(data_test))
to_val = math.floor(0.9*len(data_test))
choose = ['Bangs','Big_Nose','Bushy_Eyebrows','Chubby','Eyeglasses','Mustache']

train_generator = train_datagen.flow_from_dataframe(
    dataframe=data_test.iloc[:to_train],
    directory='/home/bruce/Downloads/CelebA/CelebA-20201202T065512Z-015/CelebA/Img/img_align_celeba/',
    x_col='images',
    y_col=choose,
    target_size=(300,300),
    class_mode='other',
    batch_size=64)

validation_generator = test_datagen.flow_from_dataframe(
    dataframe=data_test.iloc[to_train:to_val],
    directory='/home/bruce/Downloads/CelebA/CelebA-20201202T065512Z-015/CelebA/Img/img_align_celeba/',
    x_col='images',
    y_col=choose,
    target_size=(300,300),
    class_mode='other',
    batch_size=64)

# Training part
model = build_model(6)
model.compile(loss='binary_crossentropy',
              optimizer='Adadelta',
              metrics=['binary_accuracy'])

start_time = time.time()
history = model.fit_generator(train_generator,
                              steps_per_epoch=len(train_generator),
                              epochs=50,
                              validation_data=validation_generator,
                              validation_steps=len(validation_generator),
                              use_multiprocessing=True,
                              max_queue_size=128,
                              workers=8)

total_time = time.time()-start_time
print('Computational Cost: ',total_time)

save_path = args.path
save_model(model, save_path+'/model.h5')
hist_df = pd.DataFrame(history.history)
hist_df.to_csv(save_path+'/total_result.csv')
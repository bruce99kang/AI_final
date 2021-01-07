#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 14:26:10 2020

@author: bruce
"""

from tensorflow.keras import models, layers
from tensorflow.keras.applications import VGG19

import tensorflow as tf

def build_model(classes):
    model = models.Sequential()
    # Block1
    model.add(layers.Conv2D(64,(3,3),
                            activation='relu',
                            padding='same',
                            name='block1_conv1',
                            input_shape=(300,300,3)))
    model.add(layers.Conv2D(64,(3,3),
                            activation='relu',
                            padding='same',
                            name='block1_conv2'))
    model.add(layers.MaxPooling2D((2,2), strides=(2,2),name='block1_pool'))
    # Block 2
    model.add(layers.Conv2D(128,(3,3),
                            activation='relu',
                            padding='same',
                            name='block2_conv1'))
    model.add(layers.Conv2D(128,(3,3),
                            activation='relu',
                            padding='same',
                            name='block2_conv2'))
    model.add(layers.MaxPooling2D((2,2),strides=(2,2),name='block2_pool'))
    # Block3
    model.add(layers.Conv2D(256,(3,3),
                            activation='relu',
                            padding='same',
                            name='block3_conv1'))
    model.add(layers.Conv2D(256, (3,3),
                            activation='relu',
                            padding='same',
                            name='block3_conv2'))
    model.add(layers.Conv2D(256, (3,3),
                            activation='relu',
                            padding='same',
                            name='block3_conv3'))
    model.add(layers.Conv2D(256, (3,3),
                            activation='relu',
                            padding='same',
                            name='block3_conv4'))
    model.add(layers.MaxPooling2D((2,2),strides=(2,2),name='block3_pool'))
    # Block4
    model.add(layers.Conv2D(512,(3,3),
                            activation='relu',
                            padding='same',
                            name='block4_conv1'))
    model.add(layers.Conv2D(512, (3,3),
                            activation='relu',
                            padding='same',
                            name='block4_conv2'))
    model.add(layers.Conv2D(512, (3,3),
                            activation='relu',
                            padding='same',
                            name='block4_conv3'))
    model.add(layers.Conv2D(512, (3,3),
                            activation='relu',
                            padding='same',
                            name='block4_conv4'))
    model.add(layers.MaxPooling2D((2,2),strides=(2,2),name='block4_pool'))
    # Block 5
    model.add(layers.Conv2D(512,(3,3),
                            activation='relu',
                            padding='same',
                            name='block5_conv1'))
    model.add(layers.Conv2D(512, (3,3),
                            activation='relu',
                            padding='same',
                            name='block5_conv2'))
    model.add(layers.Conv2D(512, (3,3),
                            activation='relu',
                            padding='same',
                            name='block5_conv3'))
    model.add(layers.Conv2D(512, (3,3),
                            activation='relu',
                            padding='same',
                            name='block5_conv4'))
    model.add(layers.MaxPooling2D((2,2),strides=(2,2),name='block5_pool'))

    model.add(layers.Flatten(name='flatter'))
    model.add(layers.Dense(512,activation='relu',name='fc1'))
    model.add(layers.Dense(512,activation='relu',name='fc2'))
    model.add(layers.Dense(classes,activation='sigmoid',name='predictions'))

    return model

def save_model(model,path='my_model.h5'):
    model.save(path)

def load_model(path):
    model = tf.keras.models.load_model(path)
    return model

def build_pretrained(classes):

    conv_base = VGG19(include_top=False,
                      weights='imagenet',
                      input_shape=(224,224,3))
    conv_base.trainable=False
    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dense(4096,activation='relu',name='fc1'))
    model.add(layers.Dense(4096,activation='relu',name='fc2'))
    model.add(layers.Dense(classes,activation='softmax',name='predictions'))

    return model
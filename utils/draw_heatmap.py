#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 17:39:21 2020

@author: bruce
"""

import tensorflow as tf
from keras.preprocessing import image
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt

model = tf.keras.models.load_model('my_model_6class_blanced.h5')
img_path = '/home/bruce/Downloads/CelebA/CelebA-20201202T065512Z-015/CelebA/Img/img_align_celeba/199998.jpg'
img = image.load_img(img_path, target_size=(300,300))
print(type(img))
print(img.size)

x = image.img_to_array(img)
print(x.shape)
x = np.expand_dims(x, axis=0)
x = x/255.0

preds = model.predict(x)
output = model.output[:,1]
last_conv_layer = model.get_layer('block5_conv4')
grads = K.gradients(output, last_conv_layer.output)[0]
pooled_grads = K.mean(grads, axis=(0,1,2))
iterate = K.function([model.input],[pooled_grads,last_conv_layer.output[0]])

pooled_grads_value, conv_layer_output_value = iterate([x])

for i in range(512):
    conv_layer_output_value[:,:,i] *= pooled_grads_value[i]

heatmap = np.mean(conv_layer_output_value,axis=-1)
# heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
heatmap /= np.max(heatmap)

# plt.matshow(heatmap)
# plt.show()

import cv2

img = cv2.imread(img_path)
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255*heatmap)

heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
super_img = heatmap*0.4 + img
cv2.imwrite('./tmp_heatmap.jpg', super_img)
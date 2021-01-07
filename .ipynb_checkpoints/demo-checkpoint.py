#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 12:21:57 2020

@author: bruce
"""

import tensorflow as tf
import cv2
import numpy as np
import time
import argparse 
from tensorflow.keras.preprocessing.image import img_to_array
from model.model import *

# model = tf.keras.models.load_model('my_model_6class_blanced.h5')
choose = ['Bangs','Big_Nose','Bushy_Eyebrows','Chubby','Eyeglasses','Mustache']


parser = argparse.ArgumentParser()
parser.add_argument('--model', help=' find the model path', required=True, type=str,dest='model')

args = parser.parse_args()
model = tf.keras.models.load_model(args.model)
print('Loading Model finished--------------------------------')
cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    frame_g = cv2.resize(frame,(300,300))
    frame_g = frame_g.astype('float')/255.0
    frame_g = img_to_array(frame_g)
    frame_g = np.expand_dims(frame_g, axis=0)
    start_time = time.time()
    proba = model.predict(frame_g)[0]
    fps = 'FPS:' +'{:.2f}'.format(1.0/(time.time() - start_time))
    for i, element in enumerate(proba):
        label = "{}:{:.2f}%".format(choose[i],element*100)
        cv2.putText(frame, label,(10,(i*30)+25),
                    cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0))
    cv2.putText(frame, fps, (10,200),
                cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0))
    print(proba,fps)

    cv2.imshow('Result',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
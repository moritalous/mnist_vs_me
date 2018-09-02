# coding:utf-8

import keras
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import array_to_img, img_to_array,load_img
import os
import re

model = load_model('mnist_model.h5')

def list_pictures(directory, ext='jpg|jpeg|bmp|png|ppm'):
    return [os.path.join(root, f)
            for root, _, files in os.walk(directory) for f in files
            if re.match(r'([\w]+\.(?:' + ext + '))', f.lower())]

for picture in list_pictures('./tegaki/'):
    X = []
    img = img_to_array(
        load_img(picture, target_size=(28, 28), grayscale=True))
    X.append(img)

    X = np.asarray(X)
    X = X.astype('float32')
    X = X / 255.0

    features = model.predict(X)

    print('----------')
    print(picture)
    print(features.argmax())
    print('----------')

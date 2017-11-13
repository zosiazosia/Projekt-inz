from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from scipy import spatial

#jako pola base_mode?
class Transform:
    def __init__(self):
        self.base_model = VGG19(weights='imagenet')
        self.model = Model(inputs=self.base_model.input, outputs=self.base_model.get_layer('block4_pool').output)

    def transform(self, img):
        imgT  = cv2.resize(img, (224, 224))
        x = image.img_to_array(imgT)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        block4 = self.model.predict(x)
        print("shape", block4.shape)
        a = block4
        b = ((a.sum(axis=0)).mean(axis=0)).mean(axis=0)
        return b


from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import spatial

base_model = VGG19(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output)

dirInS = './Img/'
dirToClassifyS = './ImgN/'
dirIn = os.fsencode(dirInS)
dirToClassify = os.fsencode(dirToClassifyS)
#tablica do przechowywania wektorów zdjęć
tab = np.zeros((45,512))
tabC = np.zeros((14,512))
i = 0
for file in sorted(os.listdir(dirIn)):
    filename = os.fsdecode(file)


    img = image.load_img(dirInS + filename, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    block4 = model.predict(x)
    print("shape", block4.shape)
    a = block4
    b = ((a.sum(axis=0)).mean(axis=0)).mean(axis=0)
    print(i, ' '+filename)

    tab[i] = b
    i = i+1

i = 0
tree = spatial.KDTree(tab)

for file in sorted(os.listdir(dirToClassify)):
    filename = os.fsdecode(file)

    img = image.load_img(dirToClassifyS + filename, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    block4 = model.predict(x)
    print("shape", block4.shape)
    a = block4
    b = ((a.sum(axis=0)).mean(axis=0)).mean(axis=0)


    print(i," "+filename+" ",tree.query(b, k=5))
    i = i+1





from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import numpy as np
import cv2
import Posture
import Person
import matplotlib.pyplot as plt
import os
from scipy import spatial

class Transform:
    def __init__(self, id):
        self.id = id
        self.base_model = VGG19(weights='imagenet')
        self.model = Model(inputs=self.base_model.input, outputs=self.base_model.get_layer('block4_pool').output)
        self.tree = []
        self.persons = []
        self.indexes = []

    def transform(self, imgT):

        # img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(imgT)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        a = self.model.predict(x)
        b = ((a.sum(axis=0)).mean(axis=0)).mean(axis=0)
        return b

    # build tree from all currently available vectors
    def build_tree(self, persons):
        tab = []
        i = 0
        for p in persons:
            # person = p.Posture()
            for v in p.getVectors():
                tab.append(v)
                self.indexes.append(p.id)

        self.tree = spatial.KDTree(tab)

    def classify(self, persons, posture, pid):
        # first person ever, nothing to classify
        if (len(persons) == 0):
            ps = Person.Person(0)
            persons.append(ps)
            ps.addVectors(posture.getVectors())
            print("tuuuuu")

        else:
            pers = self.tree_decide(posture.getVectors())

            if (pers == 'new'):
                ps = Person.Person(len(persons))
                persons.append(ps)
                ps.addVectors(posture.getVectors())
                print("new")
            # add vectors to already existing person
            else:
                print(pers)
                persons[pers].addVectors(posture.getVectors())


    # img already as a transformed vector
    def tree_decide(self, img):
        # 5 nearest vectors
        # potem zmienić na samo img!!!!!!!!!!!
        dist, ind = self.tree.query(img[0], k=5)
        print(dist, ind)

        # new person
        if (dist[0] > 700):
            return ("new")
        # person reidentified
        else:
            return self.indexes[ind[0]]



            # better classification

    def classi(self):
        p_id = []
        i = 0
        # for v in ind:
        # nr = self.indexes[v]

        # i = i+1

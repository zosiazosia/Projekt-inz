import logging

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
import sys

import logging

logger = logging.getLogger('recognition')
hdlr = logging.FileHandler('../logs/myapp.log')
hdlr.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)



class Transform:
    def __init__(self, id, layer_name):
        self.id = id
        self.base_model = VGG19(weights='imagenet')
        self.model = Model(inputs=self.base_model.input, outputs=self.base_model.get_layer(layer_name).output)
        self.treeIn = []  # for deciding about person who's coming in
        self.treeOut = []  # for deciding about person who's coming out
        self.indexesIn = []
        self.indexesOut = []
        self.personsIn = []
        self.personsOut = []
        self.logger = logging.getLogger('recognition')
        self.logger.setLevel(logging.INFO)
        logger.info("layer_name: %s", layer_name)

    def transform(self, imgT):
        x = image.img_to_array(imgT)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        a = self.model.predict(x)
        b = ((a.sum(axis=0)).mean(axis=0)).mean(axis=0)
        return b

    # build tree from persons who are outside vectors
    def build_treeIn(self):
        tab = []
        del self.indexesIn[:]
        for p in self.personsOut:
            for v in p.getVectors():
                tab.append(v)
                self.indexesIn.append(p.id)

        self.treeIn = spatial.KDTree(tab)

    # build tree from persons who are inside vectors
    def build_treeOut(self):
        tab = []
        del self.indexesOut[:]
        for p in self.personsIn:
            for v in p.getVectors():
                tab.append(v)
                self.indexesOut.append(p.id)

        self.treeOut = spatial.KDTree(tab)

    def getIndexByPid(self, id, personsList):
        i = 0
        for p in personsList:
            if p.getId() == id:
                return i
            i = i + 1

    def classify(self, posture, counter):
        if posture.getDir() == 'in':
            if (len(self.personsOut) == 0):  # no need for building a tree here
                ps = Person.Person(len(self.personsIn))
                print("no outside")
                counter.come_in()
            else:
                self.build_treeIn()
                pers = self.tree_decideIn(posture.getVectors())
                print(pers)
                if (pers == 'new'):
                    ps = Person.Person(len(self.personsIn))
                    print("new person, but somebody is already outside")
                    counter.come_in()
                # change person localisation
                else:
                    i = self.getIndexByPid(pers, self.personsOut)
                    ps = self.personsOut.pop(i)
                    print("coming in reidentified as ", ps.getId())
                    self.logger.info('in %s', str(ps.getId()))
                    counter.reid_in()
            self.personsIn.append(ps)
            ps.addVectors(posture.getVectors())

        else:
            if (len(self.personsIn) == 0):
                print("nie może wychodzić, nikogo nie ma w środku :D")
            else:
                self.build_treeOut()
                pers = self.tree_decideOut(posture.getVectors())
                print(pers)
                i = self.getIndexByPid(pers, self.personsIn)
                ps = self.personsIn.pop(i)
                print("coming out reidentified as ", ps.getId())
                self.logger.info('out %s', str(ps.getId()))
                counter.reid_out()
                self.personsOut.append(ps)
                ps.addVectors(posture.getVectors())

    # returns person's id
    def tree_decideIn(self, vectors):
        dist, ind = self.treeIn.query(vectors[0], k=7)
        print(dist, ind)
        for i in ind:
            try:
                print(self.indexesIn[i])
            except:
                print("Not so many vectors in tree:", sys.exc_info()[0])

        # new person
        if (dist[0] > 700):
            return ("new")
        # person reidentified
        else:
            return self.indexesIn[ind[0]]

    def tree_decideOut(self, vectors):
        dist, ind = self.treeOut.query(vectors[0], k=7)
        print(dist, ind)
        for i in ind:
            try:
                print(self.indexesOut[i])
            except:
                print("Not so many vectors in tree:", sys.exc_info()[0])
        return self.indexesOut[ind[0]]


            # better classification

    def classi(self):
        p_id = []
        i = 0
        # for v in ind:
        # nr = self.indexes[v]

        # i = i+1

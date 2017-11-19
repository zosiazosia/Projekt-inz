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
        self.treeIn = []  # for deciding about person who's coming in
        self.treeOut = []  # for deciding about person who's coming out
        self.indexesIn = []
        self.indexesOut = []
        self.personsIn = []
        self.personsOut = []

    def transform(self, imgT):

        # img = image.load_img(img_path, target_size=(224, 224))
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

    def classify(self, posture, pid):

        if posture.getDir() == 'in':
            if (len(self.personsOut) == 0):  # no need for building a tree here
                ps = Person.Person(len(self.personsIn))
                print("no outside")
            else:
                self.build_treeIn()
                pers = self.tree_decideIn(posture.getVectors())
                print(pers)
                if (pers == 'new'):
                    ps = Person.Person(len(self.personsIn))
                    print("new person, but somebody is already outside")
                # change person localisation
                else:
                    i = self.getIndexByPid(pers, self.personsOut)
                    ps = self.personsOut.pop(i)
                    print("coming in reidentified as ", ps.getId())
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
                self.personsOut.append(ps)
                ps.addVectors(posture.getVectors())

    # img already as a transformed vector, returns person's id
    def tree_decideIn(self, img):
        # 5 nearest vectors
        # potem zmienić na samo img!!!!!!!!!!!
        dist, ind = self.treeIn.query(img[0], k=5)
        print(dist, ind)
        for i in ind:
            print(self.indexesIn[i])

        # new person
        if (dist[0] > 700):
            return ("new")
        # person reidentified
        else:
            return self.indexesIn[ind[0]]

    # img already as a transformed vector
    def tree_decideOut(self, img):
        # 5 nearest vectors
        # potem zmienić na samo img!!!!!!!!!!!
        dist, ind = self.treeOut.query(img[0], k=5)
        print(dist, ind)
        for i in ind:
            print(self.indexesOut[i])
        return self.indexesOut[ind[0]]


            # better classification

    def classi(self):
        p_id = []
        i = 0
        # for v in ind:
        # nr = self.indexes[v]

        # i = i+1

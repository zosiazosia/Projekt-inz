import logging
from collections import Counter as pyCounter
import operator
from keras.applications.vgg19 import VGG19
from keras.backend import clear_session
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.applications.resnet50 import ResNet50
from keras.models import Model
import numpy as np
import Person
from scipy import spatial
import sys
import tensorflow
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
        # tensorflow.keras.backend.clear_session()
        clear_session()
        self.base_model = VGG19(weights='imagenet')
        self.model = Model(inputs=self.base_model.input, outputs=self.base_model.get_layer(layer_name).output)
        self.treeIn = []  # for deciding about person who's coming in
        self.treeOut = []  # for deciding about person who's coming out
        self.indexesIn = []
        self.indexesOut = []
        self.personsIn = []
        self.personsOut = []
        self.distThreshold = 2200  #threshold to decide if it is a new person
        self.logger = logging.getLogger('recognition')
        self.logger.setLevel(logging.INFO)
        logger.info("layer_name: %s", layer_name)

    def transform(self, imgT):
        x = image.img_to_array(imgT)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # wektor 'a' ma wymiary 1:14:14:512
        a = self.model.predict(x)

        # wektor 'ret1' to wersja podstawowa - skumulowanie
        # czterowymiarowego wektora x do jednowymiarowego, o wymiarze 512
        ret1 = ((a.sum(axis=0)).mean(axis=0)).mean(axis=0)

        # wektor 'ret' to wersja przy podziale wektora x na 3 czesci, rozmiar 512*3 = 1536
        up = (a[0][0:4].mean(axis=0)).mean(axis=0)
        mid = (a[0][4:10].mean(axis=0)).mean(axis=0)
        down = (a[0][10:14].mean(axis=0)).mean(axis=0)
        ret = np.concatenate((up, mid, down), axis=0)
        # wektor 'ret2' to wersja druga podziału wektora x na 3 części, rozmiar 512*3 = 1536
        up2 = ((a[0].mean(axis=0))[0:4]).mean(axis=0)
        mid2 = ((a[0].mean(axis=0))[4:10]).mean(axis=0)
        down2 = ((a[0].mean(axis=0))[10:14]).mean(axis=0)
        ret2 = np.concatenate((up2, mid2, down2), axis=0)
        return ret

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
                pers = self.tree_decide(posture.getVectors(), 'in')
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
                pers = self.tree_decide(posture.getVectors(), 'out')
                print(pers)
                i = self.getIndexByPid(pers, self.personsIn)
                ps = self.personsIn.pop(i)
                print("coming out reidentified as ", ps.getId())
                self.logger.info('out %s', str(ps.getId()))
                counter.reid_out()
                self.personsOut.append(ps)
                ps.addVectors(posture.getVectors())

    # returns person's id or "new"
    def tree_decide(self, vectors, direction):

        if direction == 'in':
            tree = self.treeIn
            indexes = self.indexesIn
        elif direction == 'out':
            tree = self.treeOut
            indexes = self.indexesOut

        # return self.mostFreqNearest(vectors, tree, indexes, direction)
        return self.kMultiplyDistance(vectors, tree, indexes, direction, 5)

    def mostFreqNearest(self, vectors, tree, indexes, direction):
        print("decyzja dla osoby")
        nearests = []
        minD = self.distThreshold
        for vector in vectors:
            dist, ind = tree.query(vector, k=10)  # k-nearest vectors
            nearests.append(indexes[ind[0]])

            if dist[0] < minD:
                minD = dist[0]

            # to potem do usunięcia - tylko w celach testowych
            # dist - odległość, ind - określenie miejsca w drzewie, indexes określają id osoby do której przynależy wektor
            print(dist, ind)
            for i in ind:
                try:
                    print(indexes[i])
                except:
                    print("Not so many vectors in tree:", sys.exc_info()[0])

        # new person coming in
        if direction == 'in':
            if minD == self.distThreshold:
                return "new"

        print("najczęściej ", self.mostFrequent(nearests))
        return self.mostFrequent(nearests)

    def kMultiplyDistance(self, vectors, tree, indexes, direction, k):

        print("decyzja dla osoby")
        nearests = []
        dictN = {}
        minD = self.distThreshold
        for vector in vectors:
            dist, ind = tree.query(vector, k=k)  # k-nearest vectors
            for x in range(0, k):
                if indexes[ind[x]] not in dictN:
                    dictN[indexes[ind[x]]] = (dist[x], 1)
                else:
                    dictN[indexes[ind[x]]] = (dictN[indexes[ind[x]]][0] + dist[x], dictN[indexes[ind[x]]][1] + 1)

            if dist[0] < minD:
                minD = dist[0]

            # to potem do usunięcia - tylko w celach testowych
            # dist - odległość, ind - określenie miejsca w drzewie, indexes określają id osoby do której przynależy wektor
            print(dist, ind)
            for i in ind:
                try:
                    print(indexes[i])
                except:
                    print("Not so many vectors in tree:", sys.exc_info()[0])

        # new person coming in
        if direction == 'in':
            if minD == self.distThreshold:
                return "new"

        for key, value in dictN.items():
            dictN[key] = value[0] / value[1]
        print("średnie ", dictN)
        nearest = sorted(dictN.items(), key=operator.itemgetter(1))[0][0]

        print("najmniejsza średnia: ", nearest)
        return nearest

    def mostFrequent(self, nearests):
        freqDict = dict(pyCounter(nearests))
        sortFreq = sorted(freqDict.items(), key=operator.itemgetter(1))
        return sortFreq[-1][0]

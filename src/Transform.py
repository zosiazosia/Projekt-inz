import logging
from collections import Counter as pyCounter
import operator
from keras.applications.vgg19 import VGG19
from keras.backend import clear_session
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
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
        clear_session()
        self.base_model = VGG19(weights='imagenet')
        self.model = Model(inputs=self.base_model.input, outputs=self.base_model.get_layer(layer_name).output)
        self.treeIn = []  # for deciding about person who's coming in
        self.treeOut = []  # for deciding about person who's coming out
        self.indexesIn = []
        self.indexesOut = []
        self.personsIn = []
        self.personsOut = []
        self.distThreshold = 465  # threshold to decide if it is a new person, depends on dataset!
        self.logger = logging.getLogger('recognition')
        self.logger.setLevel(logging.INFO)
        logger.info("layer_name: %s", layer_name)

    def transform(self, imgT):
        x = image.img_to_array(imgT)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # vector 'a' is size 1:14:14:512 for 'block5_conv2' layer
        a = self.model.predict(x)

        # vector 'ret'  - divides 'a' into three parts and then join them (final size - 1536)
        up = (a[0][0:4].mean(axis=0)).mean(axis=0)
        mid = (a[0][4:10].mean(axis=0)).mean(axis=0)
        down = (a[0][10:14].mean(axis=0)).mean(axis=0)
        ret = np.concatenate((up, mid, down), axis=0)

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
            counter.increase_regular_in()
            if len(self.personsOut) == 0:  # no need for building a tree here
                ps = Person.Person(len(self.personsIn))
                print("coming in as new ", ps.getId())
            else:
                self.build_treeIn()
                pers = self.tree_decide(posture.getVectors(), 'in')
                if pers == 'new':
                    ps = Person.Person(len(self.personsIn) + len(self.personsOut))
                    print("coming in as new ", ps.getId())
                else:  # change person localisation
                    i = self.getIndexByPid(pers, self.personsOut)
                    ps = self.personsOut.pop(i)
                    print("coming in reidentified as ", ps.getId())
                    self.logger.info('in %s', str(ps.getId()))
                    counter.reid_in()
            self.personsIn.append(ps)
            ps.addVectors(posture.getVectors())

        else:
            counter.increase_regular_out()
            if len(self.personsIn) == 0:
                counter.error_information = "Wykryto osobę wychodzącą pomimo, że pomieszczenie jest puste. "
            else:
                self.build_treeOut()
                pers = self.tree_decide(posture.getVectors(), 'out')
                if pers == 'new':
                    ps = Person.Person(len(self.personsIn) + len(self.personsOut))
                    print("coming out as new ", ps.getId())
                else:  # change person localisation
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

        # different functions for deciding about which person is the most similar
        return self.mostFreqNearest(vectors, tree, indexes)
        # return self.kMultiplyDistance(vectors, tree, indexes, 5)

    def mostFreqNearest(self, vectors, tree, indexes):
        nearests = []
        minD = self.distThreshold
        for vector in vectors:
            dist, ind = tree.query(vector, k=10)  # k-nearest vectors
            nearests.append(indexes[ind[0]])

            if dist[0] < minD:
                minD = dist[0]

        # not similar enough to anyone identified before
        if minD == self.distThreshold:
            return "new"

        return self.mostFrequent(nearests)

    def kMultiplyDistance(self, vectors, tree, indexes, k):

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

        # not similar enough to anyone identified before
        if minD == self.distThreshold:
            return "new"

        for key, value in dictN.items():
            dictN[key] = value[0] / value[1]
        nearest = sorted(dictN.items(), key=operator.itemgetter(1))[0][0]

        return nearest

    def mostFrequent(self, nearests):
        freqDict = dict(pyCounter(nearests))
        sortFreq = sorted(freqDict.items(), key=operator.itemgetter(1))
        return sortFreq[-1][0]

from random import randint
import time

class MyPerson:
    tracks = []
    def __init__(self, i, xi, yi, max_age):
        self.i = i
        self.x = xi
        self.y = yi
        self.tracks = []
        self.R = randint(0,255)
        self.G = randint(0,255)
        self.B = randint(0,255)
        self.done = False
        self.state = '0'
        self.age = 0
        self.max_age = max_age
        self.dir = None
        self.lastTime = int(time.strftime("%S"))
        self.vectors = []
    def getRGB(self):
        return (self.R,self.G,self.B)
    def addVector(self, vector):
        self.vectors.append(vector)
    def getTracks(self):
        return self.tracks
    def getVectors(self):
        return self.vectors
    def getId(self):
        return self.i
    def getState(self):
        return self.state
    def getDir(self):
        return self.dir
    def getX(self):
        return self.x
    def getY(self):
        return self.y
    def updateCoords(self, xn, yn):
        self.age = 0
        self.tracks.append([self.x,self.y])
        self.x = xn
        self.y = yn
        self.lastTime = int(time.strftime("%S"))
        if len(self.tracks) >= 2:
            if self.tracks[-2][0] < self.tracks[-1][0]:
                self.dir = 'left'
            else:
                self.dir = 'right'
    def getLastTime(self):
        return self.lastTime
    def setDone(self):
        self.done = True
    def timedOut(self):
        return self.done

    def going_LEFT(self, mid_end):
        if len(self.tracks) >= 3:
            if self.state == '0':
                if self.tracks[-1][0] < mid_end and self.tracks[-2][0] >= mid_end:  # cruzo la linea
                    state = '1'
                    self.dir = 'left'
                    return True
            else:
                return False
        else:
            return False

    def going_RIGHT(self, mid_start):
        if len(self.tracks) >= 3:
            if self.state == '0':
                if self.tracks[-1][0] > mid_start and self.tracks[-2][0] <= mid_start:  #cruzo la linea
                    state = '1'
                    self.dir = 'right'
                    return True
            else:
                return False
        else:
            return False
    def age_one(self):
        self.age += 1
        if self.age > self.max_age:
            self.done = True
        return True
        # class MultiPerson:
        #     def __init__(self, persons, xi, yi):
        #         self.persons = persons
        #         self.x = xi
        #         self.y = yi
        #         self.tracks = []
        #         self.R = randint(0,255)
        #         self.G = randint(0,255)
        #         self.B = randint(0,255)
        #         self.done = False
        #

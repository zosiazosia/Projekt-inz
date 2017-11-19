from random import randint
import time
import Counter


class Posture:
    tracks = []

    def __init__(self, id, xi, yi):
        self.id = id
        self.x = xi
        self.y = yi
        self.tracks = []
        self.R = randint(0, 255)
        self.G = randint(0, 255)
        self.B = randint(0, 255)
        self.state = '0'
        self.dir = None
        self.lastTime = int(time.strftime("%M%S"))
        self.vectors = []

    def getRGB(self):
        return (self.R, self.G, self.B)

    def addVector(self, vector):
        self.vectors.append(vector)

    def getTracks(self):
        return self.tracks

    def getVectors(self):
        return self.vectors

    def getId(self):
        return self.id

    def getState(self):
        return self.state

    def getDir(self):
        return self.dir

    def getX(self):
        return self.x

    def getY(self):
        return self.y

    def addCoords(self, xn, yn):
        self.tracks.append([self.x, self.y])
        self.x = xn
        self.y = yn
        self.lastTime = int(time.strftime("%M%S"))
        if len(self.tracks) >= 2:
            if self.tracks[-2][0] < self.tracks[-1][0]:
                self.dir = 'left'
            else:
                self.dir = 'right'

    def getLastTime(self):
        return self.lastTime

    def going_IN(self, line_left, line_right, counter):
        if len(self.tracks) >= 3:
            if self.state == '0':
                if counter.getInDirection() == 'left':
                    if (self.tracks[-1][0] < line_right and self.tracks[-2][0] >= line_right):
                        self.state = '1'
                        self.dir = 'in'
                        return True
                elif counter.getInDirection() == 'right':
                    if (self.tracks[-1][0] > line_left and self.tracks[-2][0] <= line_left):
                        self.state = '1'
                        self.dir = 'in'
                        return True
            else:
                return False
        else:
            return False

    def going_OUT(self, line_left, line_right, counter):
        if len(self.tracks) >= 3:
            if self.state == '0':
                if counter.getInDirection() == 'left':
                    if (self.tracks[-1][0] > line_left and self.tracks[-2][0] <= line_left):
                        self.state = '1'
                        self.dir = 'out'
                        return True
                elif counter.getInDirection() == 'right':
                    if (self.tracks[-1][0] < line_right and self.tracks[-2][0] >= line_right):
                        self.state = '1'
                        self.dir = 'out'
                        return True
            else:
                return False
        else:
            return False

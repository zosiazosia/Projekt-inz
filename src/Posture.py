from random import randint
import time


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
        self.vectorSaved = False
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

    def getVectorSaved(self):
        return self.vectorSaved

    def setVectorSaved(self, a):
        self.vectorSaved = a

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

    def going_LEFT(self, line_left, line_right):
        if len(self.tracks) >= 3:
            if self.state == '0':
                # aktualne położenie jest mniejsze niż lewa a poprzednie było po prawej stronie lewej linii
                # if ((self.tracks[-1][0] < line_left and self.tracks[-2][0] >= line_left) or (self.tracks[-1][0] < line_right and self.tracks[-2][0] >= line_right)) :
                if (self.tracks[-1][0] < line_right and self.tracks[-2][0] >= line_right):
                    self.state = '1'
                    self.dir = 'left'
                    return True
            else:
                return False
        else:
            return False

    def going_RIGHT(self, line_left, line_right):
        if len(self.tracks) >= 3:
            if self.state == '0':
                # if ((self.tracks[-1][0] > line_right and self.tracks[-2][0] <= line_right) or (self.tracks[-1][0] > line_left and self.tracks[-2][0] <= line_left)):
                if (self.tracks[-1][0] > line_left and self.tracks[-2][0] <= line_left):
                    self.state = '1'
                    self.dir = 'right'
                    return True
            else:
                return False
        else:
            return False

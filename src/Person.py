class Person:
    def __init__(self, id):
        self.vectors = []
        self.id = id
        self.inRoom = True

    def addVector(self, vector):
        self.vectors.append(vector)

    def addVectors(self, vectors):
        self.vectors.extend(vectors)

    def getVectors(self):
        return self.vectors

    def getId(self):
        return self.id

    def getInRoom(self):
        return self.inRoom

    def setInRoom(self, inRoom):
        self.inRoom = inRoom

class Person:
    def __init__(self, id):
        self.vectors = []
        self.id = id

    def addVector(self, vector):
        self.vectors.append(vector)

    def getVectors(self):
        return self.vectors
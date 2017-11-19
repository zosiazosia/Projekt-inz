class Person:
    def __init__(self, _id):
        self.vectors = []
        self._id = _id

    def addVector(self, vector):
        self.vectors.append(vector)

    def addVectors(self, vectors):
        self.vectors.extend(vectors)

    def getVectors(self):
        return self.vectors

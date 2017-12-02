class Counter:
    def __init__(self, inDir):
        self.came_in = 0
        self.came_out = 0
        self.reident_in = 0
        self.reident_out = 0
        self.are_inside = 0
        self.inDirection = inDir  # 'left or 'right'
        self.regular_left = 0
        self.regular_right = 0

    def come_in(self):
        self.came_in += 1
        self.are_inside += 1

    def come_out(self):
        self.came_out += 1
        self.are_inside -= 1

    def reid_in(self):
        self.reident_in += 1
        self.are_inside += 1

    def reid_out(self):
        self.reident_out += 1
        self.are_inside -= 1

    def getCameIn(self):
        return self.came_in

    def getCameOut(self):
        return self.came_out

    def getReidentIn(self):
        return self.reident_in

    def getReidentOut(self):
        return self.reident_out

    def getAreInside(self):
        return self.are_inside

    def getInDirection(self):
        return self.inDirection

    def generate_report(self):
        info = "came_in: " + str(self.getCameIn()) + ", came_out: " + str(self.getCameOut()) + ", reid_in: " + str(
            self.getReidentIn()) + ", reid_out: " + str(self.getReidentOut()) + ", inside: " + str(self.getAreInside())
        return info

    def increase_regular_left(self):
        self.regular_left += 1

    def increase_regular_right(self):
        self.regular_right += 1

class Counter:
    def __init__(self):
        self.came_in = 0
        self.came_out = 0
        self.reident_in = 0
        self.reident_out = 0
        self.are_inside = 0

    def come_in(self):
        self.came_in += 1
        self.are_inside += 1

    def come_out(self):
        self.came_out += 1
        self.are_inside -= 1

    def reident_in(self):
        self.reident_in += 1
        self.are_inside += 1

    def reident_out(self):
        self.reident_out += 1
        self.are_inside -= 1

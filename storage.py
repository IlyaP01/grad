from numpy import array


class Storage:
    def __init__(self):
        self.trace = []

    def add(self, point: array):
        self.trace.append(point)

    def get_trace(self):
        return self.trace

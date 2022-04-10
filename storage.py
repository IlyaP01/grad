from numpy import array

class Storage:
    def __init__(self):
        self.trace = []


    def add(self, point: array, vec: array):
        self.trace.append((point, vec))


    def get_trace(self):
        return self.trace
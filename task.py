import math
from numpy import array

class Task:
    def __init__(self):
        self.count = 0


    @staticmethod
    def f(x: array):
        return x[0] * x[0] + x[1] * x[1] + math.cos(2 * x[0] + 3 * x[1])


    def f_count(self, x: array):
        self.count += 1
        return Task.f(x)


    def get_count(self):
        return self.count

    @staticmethod
    def initial_guess():
        return array([0, 0])


task = Task()
print(task.f_count(array([0, 0])))
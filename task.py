from numpy import array
from numpy import cos, sin

class Task:
    def __init__(self):
        self.count = 0
        self.count1 = 0


    @staticmethod
    def f(x: array):
        return x[0] * x[0] + x[1] * x[1] + cos(2 * x[0] + 3 * x[1])


    def f_count(self, x: array):
        self.count += 1
        return Task.f(x)


    @staticmethod
    def grad_f(x: array):
        sinus = sin(2 * x[0] + 3 * x[1])
        return array([2 * x[0] - 2 * sinus, 2 * x[1] - 3 * sinus])


    def grad_f_count(self, x: array):
        self.count1 += 1
        return Task.grad_f(x)


    def get_count(self):
        return self.count


    def get_grad_count(self):
        return self.count1


    @staticmethod
    def initial_guess():
        return array([1, 0])


    @staticmethod
    def grad1_params():
        return (0.5, 0.7, 0.5)


    @staticmethod
    def get_accuracies():
        return [0.1, 0.01, 0.001]

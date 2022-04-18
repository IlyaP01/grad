from task import Task
from storage import Storage

import numpy as np


def solve(task: Task, storage: Storage, eps):
    x = task.initial_guess()
    alpha0, _lambda, delta = task.grad1_params()
    while True:
        grad = task.grad_f_count(x)
        grad_norm = np.linalg.norm(grad, ord=2)
        if grad_norm < eps:
            break

        fk = task.f_count(x)
        storage.add(np.array([x[0], x[1], fk]))
        alpha = alpha0
        grad_norm_2 = grad_norm ** 2
        while task.f_count(x - alpha * grad) - fk > -delta * alpha * grad_norm_2:
            alpha *= _lambda

        x = x - alpha * grad

    return x, fk

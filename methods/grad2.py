from task import Task
from storage import Storage
import numpy as np
from . import golden_section


def find_alpha(f, x, p, eps):
    return golden_section.solve(lambda alpha: f(x + alpha * p), 0, 1, eps)


def calc_A(A, dx, dw):
    s = dw.dot(dx)
    t = np.dot(dw, A.dot(dw))
    r = A.dot(dw) / t - dx / s
    return A - np.outer(dx, dx) / s - A.dot(np.outer(dw, dw)).dot(A.T) / t + t * np.outer(r, r)


def solve(task: Task, storage: Storage, eps):
    x_prev = task.initial_guess()
    n = len(x_prev)
    A = np.eye(n)
    w_prev = -task.grad_f_count(x_prev)
    storage.add(np.array([x_prev[0], x_prev[1], task.f(x_prev)]))

    while np.linalg.norm(w_prev) >= eps:
        p = A.dot(w_prev)
        alpha = find_alpha(task.f_count, x_prev, p, eps)
        x = x_prev + alpha * p
        w = -task.grad_f_count(x)
        dx = x - x_prev
        dw = w - w_prev
        A = calc_A(A, dx, dw)
        x_prev = x
        w_prev = w
        storage.add(np.array([x_prev[0], x_prev[1], task.f(x_prev)]))

    return x_prev, task.f(x_prev)

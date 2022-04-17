from pandas import array
from task import Task
from storage import Storage
from numpy import linalg
from numpy import array

def solve(task: Task, storage: Storage, eps):
    x = task.initial_guess()
    f = task.f_count(x)
    (l, v, t) = task.grad1_params()
    while True:
        grad = task.grad_f_count(x)
        norm = linalg.norm(grad, ord=2)
        s = -grad / norm
        storage.add(array([x[0], x[1], f]))
        x = x + l * s
        f1 = task.f_count(x)
        delta_f = f - f1
        f = f1
        if l <= eps:
            break
        while abs(delta_f) < t * l * norm:
            l = v * l
            continue
    return (x, f)
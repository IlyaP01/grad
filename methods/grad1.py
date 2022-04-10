from task import Task
from storage import Storage
from numpy import linalg

def solve(task: Task, storage: Storage, eps):
    x = task.initial_guess()
    f = task.f_count(x)
    (l, v) = task.grad1_params()
    while True:
        grad = task.grad_f_count(x)
        norm = linalg.norm(grad, ord=2)
        s = -grad / norm
        storage.add(x, l * s)
        x = x + l * s
        f1 = task.f_count(x)
        delta_f = f - f1
        f = f1
        if delta_f >= 0.5 * l * norm:
            l = v * l
            continue
        if abs(delta_f) <= eps:
            break
    return (x, f)
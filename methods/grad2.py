from task import Task
from storage import Storage
import numpy as np


def solve(task: Task, storage: Storage, eps):
    x = task.initial_guess()
    n = len(x)
    A = np.eye(n)
    ...


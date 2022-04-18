from task import Task
from storage import Storage
from methods import grad1, grad2

import matplotlib.pyplot as plt

import numpy as np

fig, ax = plt.subplots()
X = np.arange(-0.5, 2, 0.1)
Y = np.arange(-0.5, 2, 0.1)
X, Y = np.meshgrid(X, Y)
Z = Task.f(np.array([X, Y]))
cs = ax.contour(X, Y, Z, levels=30)
ax.clabel(cs, inline=1, fontsize=10)

task = Task()
eps = 0.1
storage1 = Storage()
storage2 = Storage()
grad1.solve(task, storage1, eps)
grad2.solve(task, storage2, eps)
trace1 = (storage1.get_trace())
trace2 = (storage2.get_trace())

plt.plot([x[0] for x in trace1], [x[1] for x in trace1], label='grad1')
plt.plot([x[0] for x in trace1], [x[1] for x in trace1], '*')

plt.plot([x[0] for x in trace2], [x[1] for x in trace2], label='BFGS')
plt.plot([x[0] for x in trace2], [x[1] for x in trace2], '*')

plt.legend()
plt.show()

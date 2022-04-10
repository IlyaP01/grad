from task import Task

import matplotlib.pyplot as plt
from matplotlib import cm

import numpy as np

fig = plt.figure()
ax = fig.gca(projection='3d')
X = np.arange(-2, 2, 0.1)
Y = np.arange(-2, 2, 0.1)
X, Y = np.meshgrid(X, Y)
Z = Task.f(np.array([X, Y]))
cset = ax.contour(X, Y, Z, cmap=cm.coolwarm)
ax.clabel(cset, fontsize=9, inline=1)

plt.show()
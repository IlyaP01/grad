from task import Task

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

import csv


fig = plt.figure()
ax = fig.gca(projection='3d')
X = np.arange(0, 1, 0.1)
Y = np.arange(0, 1, 0.1)
X, Y = np.meshgrid(X, Y)
Z = Task.f(np.array([X, Y]))
cset = ax.contour(X, Y, Z, cmap=cm.coolwarm)
ax.clabel(cset, fontsize=9, inline=1)

with open('data/grad1_0_1.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    prev = [float(coord) for coord in next(reader)]
    for record in reader:
        cur = [float(coord) for coord in record]
        ax.plot([prev[0], cur[0]], [prev[1], cur[1]], [prev[2], cur[2]], '-k', linewidth=1)
        print(prev, cur)
        prev = cur

plt.draw()
plt.show()

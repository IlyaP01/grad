from task import Task
from storage import Storage
from methods import grad1

import csv


def solve(name, solver):
    for eps in Task.get_accuracies():
        task = Task()
        storage = Storage()
        (x, f) = grad1.solve(task, storage, eps)
        print("Eps: " + str(eps) + ", x: " + str(x) + ", f(x): " + str(f))
        print("Function calls: " + str(task.get_count()) + ", grad calls: " + str(task.get_grad_count()))
        with open("data/" + name + '_' + str(eps).replace('.', '_') + '.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for point in storage.get_trace():
                writer.writerow(point)
            writer.writerow([x[0], x[1], f])


print("First order:")
solve("grad1", grad1.solve)

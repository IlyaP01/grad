from task import Task
from storage import Storage
from methods import grad1


def solve(solver):
    for eps in Task.get_accuracies():
        task = Task()
        storage = Storage()
        (x, f) = grad1.solve(task, storage, eps)
        print("Eps: " + str(eps) + ", x: " + str(x) + ", f(x): " + str(f))
        print("Function calls: " + str(task.get_count()) + ", grad calls: " + str(task.get_grad_count()))
        print("Points: ", end="")
        # for el in storage.get_trace():
        #     print(str(el[0]), end=", ")
        # print()


print("First order:")
solve(grad1.solve)

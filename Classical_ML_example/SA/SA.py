'''
A simple example of Simulated Annealing algorithm for searching minimum of f(x) = 3*x**2 - 60*x + 9 with x in [0,100]. The anwser is 10.
'''
import numpy as np
from matplotlib import pyplot as plt
def x_function(x):
    return 3*x**2 - 60*x + 9
 
x = [i for i in np.linspace(0, 100)]
y = map(x_function, x)
plt.plot(x, list(y))
plt.show()
# 初始温度
T = 1
x = np.random.uniform(0, 100)
# 终止温度
std = 0.00000001
# 衰减率
a = 0.999
while T > std:
    y = x_function(x)
    # 新值通过扰动产生
    x_new = x + np.random.uniform(-1, 1)
    if 0 <= x_new <= 100:
        y_new = x_function(x_new)
        if y_new < y:
            x = x_new
        else:
            p = np.exp((y - y_new) / T)
            r = np.random.uniform(0, 1)
            if p > r:
                x = x_new
                # print(x)
    T = T * a
print(x, x_function(x))
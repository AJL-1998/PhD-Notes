import numpy as np
import matplotlib.pyplot as plt

n = 100
var = 10000
epsilon = np.sqrt(var)*np.random.normal(size = (n,1))

def model_function(x):
    return 2 + x + x ** 2 - 0.01*x ** 3
x_vals = np.array([i for i in range(n)])
y_vals = model_function(np.vstack(x_vals)) + epsilon
num_params = 4
X = np.zeros((n,num_params))
for i in range(n):
    for j in range(num_params):
        X[i,j] = x_vals[i] ** j
Xt = X.T
Xhat = Xt @ X
H = np.linalg.inv(Xhat) @ Xt @ y_vals
print(H)
Y = X @ H
plt.figure()
plt.scatter(x_vals, y_vals, marker = '.')
plt.plot(Y, color = 'red')
plt.show()
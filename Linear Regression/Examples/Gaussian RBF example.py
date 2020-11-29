import numpy as np
import matplotlib.pyplot as plt

d = 10
N = 20
N_star = 100
x = np.linspace(0.0, 10.0, N)
x_star = np.linspace(0.0, 10.0, N_star)
y = np.vstack(x * np.sin(0.5*x ** 2))

centres = np.array([i for i in range(d)])
X = np.zeros((N, d))
for i in range(N):
    for j in range(d):
        X[i,j] = np.exp(-0.5 * (x[i] - centres[j]) ** 2)
Xt = X.T
H = Xt @ X
H = np.linalg.inv(H)
params = H @ Xt @ y
y_star = np.zeros((N_star))
def kernel(x, d):
    f = np.array([np.exp(-0.5* (x - i) ** 2) for i in d])
    return np.vstack(f)
for i in range(N_star):
    y_star[i] = params.T @ kernel(x_star[i], centres)
plt.figure()
plt.plot(x_star, y_star, color = 'red')
plt.scatter(x, y, marker = '.')
plt.show()
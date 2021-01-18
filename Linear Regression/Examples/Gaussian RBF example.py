import numpy as np
import matplotlib.pyplot as plt

d = 20
N = 19
N_star = 1000
x   =  np.array([i for i in range(1,20)]) 
y   = np.array([3,4,3,10,9,5,8,2,4,4,16,8,10,14,6,7,18,40,36])
x_star = np.linspace(-10, 10.0, N_star)

centres = np.array([i for i in range(1,20)])
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
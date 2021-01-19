import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(6)

ls = np.array([[0.2,0], [0,0.2]])
ARD = np.linalg.inv(ls)
l1 = ARD[0,0]
l2 = ARD[1,1]
nx = 30
x_min = 0
x_max = 1
x = np.linspace(x_min,x_max, nx)
X, Y = np.meshgrid(x,x)
Z = np.array([X,Y]).T
K = []
for i in range(nx):
    for j in range(nx):
        for k in range(nx):
            for l in range(nx):
                coef = Z[i,j,:] - Z[k,l,:]
                coef = (coef[0] ** 2) / l1 **2 + (coef[1]) ** 2 / l2 ** 2
                K.append(coef)

sigma = 0.3
l = 0.7
K = np.reshape(K, (nx**2,nx**2))
K = np.exp( - 0.5 * K**2 ) + sigma ** 2 *np.eye(nx**2)
y = np.random.multivariate_normal([0 for i in range(nx**2)], K)
y = np.reshape(y, (nx,nx))

'''fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, y, cmap=cm.viridis)
ax.set_xlim(x_min,x_max)
ax.set_xlim(x_min,x_max)

# Adjust the limits, ticks and view angle
ax.view_init(25, 30)
plt.savefig('hyper_2d')

plt.figure(figsize=(5,5))
plt.imshow(y, interpolation='bilinear', origin='lower')	
plt.colorbar()
plt.xticks([])
plt.yticks([])
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()'''
print(np.linalg.eigvals(K))


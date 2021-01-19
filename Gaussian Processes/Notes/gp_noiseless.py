import numpy as np
import matplotlib.pyplot as plt
from scipy import spatial as spt
from matplotlib import cm

np.random.seed(98765)

def find_s_d(X1, X2 = None):
    if X2 is not None:
        sqd = spt.distance.cdist(X1, X2, metric='sqeuclidean')
    else:
        sqd = spt.distance.cdist(X1, X1, metric='sqeuclidean')
    return sqd

def SE(X1, X2, l=1., sigma_f=1.):
    sqdist = find_s_d(X1, X2)
    return sigma_f**2 * np.exp(-0.5 / l**2 * sqdist)

def Per(X1, X2, l = 1., p = 1., sigma_f = 1.):
    sqdist = find_s_d(X1, X2)
    return sigma_f ** 2 * np.exp(- 2 * l **(-2) * np.sin(np.pi * np.sqrt(sqdist) / p) ** 2)

def RQ(X1, X2, l =1., alpha = 1., sigma = 1.):
    sqdist = find_s_d(X1, X2)
    return ( 1 + sqdist / (2 * alpha * l ** 2)) ** (-1 * alpha)

def LinKer(X1, X2, c = 1., d = 1. ):
    return (X1 - c) @ (X2 - d).T

x = np.vstack( np.random.uniform(-10,10,10) )
x_s = np.vstack( np.linspace(-10,10,200))
y = x * np.sin(x) ** 2
K = SE(x,x, l =1., sigma_f = 1.) + Per(x,x, l =2., p=2.)
K_s = SE(x_s,x) + Per(x_s,x, l =2., p=2.)
K_ss = SE(x_s,x_s) + Per(x_s,x_s, l =2., p=2.)

mu_s = K_s.dot(np.linalg.inv(K)).dot(y)
mu_s = np.hstack(mu_s).tolist()
cov_s = K_ss - K_s.dot(np.linalg.inv(K)).dot(K_s.T)

y_draw = np.random.multivariate_normal(mu_s, cov_s)

y_true = x_s * np.sin(x_s) ** 2

cov_p = mu_s + 2*cov_s.diagonal()
cov_m = mu_s - 2*cov_s.diagonal()


plt.figure(figsize = (12,4))
plt.subplot(121)
plt.plot(x_s, y_true, 'm--', alpha=0.4)
plt.plot(x_s, [0 for i in x_s], color = 'black', alpha = 0.6, linestyle=(0, (3, 1, 1, 1, 1, 1)), label = 'Prior Mean')
plt.fill_between(np.linspace(-10,10,200), -2, 2, alpha = 0.3, label = 'Prior Variance')
plt.plot(x_s, np.random.multivariate_normal([0 for i in x_s], K_ss), label='Sample Function')
plt.ylim([-10,10])
plt.xticks([])
plt.yticks([])
plt.legend(loc=0, fontsize = 5)
plt.legend(loc=0)

plt.subplot(122)
plt.scatter(x,y, marker='x', c='Black', label = 'Observed data')
plt.plot(x_s, mu_s, color='black', alpha=0.6, linestyle=(0, (3, 1, 1, 1, 1, 1)), label = 'Predictive Mean')
plt.plot(x_s, y_true, 'm--', alpha=0.4, label = 'Underlying function')
plt.plot(x_s, y_draw, label = 'Sample function')
plt.fill_between(np.linspace(-10,10,200), cov_m, cov_p, alpha = 0.3, label = 'Predictive Variance')
plt.ylim([-10,10])
plt.xticks([])
plt.yticks([])
plt.legend(loc=0, fontsize = 7)
plt.savefig('gp_ex.pdf')
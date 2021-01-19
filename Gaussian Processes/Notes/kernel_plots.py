import numpy as np
import matplotlib.pyplot as plt
from scipy import spatial as spt
from matplotlib import cm

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

x = np.vstack(np.linspace(0.0,20.0, 1000))
x_r = np.linspace(-10.,10, 1000)
y_Per1 = 0.25 * np.exp(- 2 * 0.5 **(-2) * np.cos(np.pi * x_r  / 1.) ** 2)
y_Per2 = 0.25 * np.exp(- 2 * 3.0 **(-2) * np.cos(np.pi * x_r  / 5.5) ** 2)
y_RQ1 = 0.25*( 1 + x_r ** 2 / (2 * 0.5 * 0.5 ** 2)) ** (-1 * 0.5)
y_RQ2 = 0.25*( 1 + x_r ** 2 / (2 * 2.5 * 1.5 ** 2)) ** (-1 * 2.5)
mu = np.array([0 for i in x])
K11 = Per(x,x, l = 1.5, p = 1.5, sigma_f = 0.5)
K12 = Per(x,x, l = 3., p = 5., sigma_f = 0.5)
K21 = RQ(x,x,l=0.5, alpha = 0.5, sigma = 0.5)
K22 = RQ(x,x,l=1.5, alpha = 2.5, sigma = 0.5)



y1 = np.random.multivariate_normal(mu, K11)
y2 = np.random.multivariate_normal(mu, K12)
plt.figure(figsize = (12,4))
plt.subplot(121)
plt.plot(x_r, y_Per1, label = '$l = 1.5, p = 1.5$')
plt.plot(x_r, y_Per2, label = '$l = 3.0, p = 5.0$')
plt.xlim([-5,5])
plt.ylim([-0.01,0.3])
plt.yticks([0,0.25], labels=[0,'$\sigma^2$'])
plt.xticks([0])
plt.xlabel('$x - x^{\prime}$')
plt.ylabel('$\kappa(x,x^{\prime}) $')
plt.legend(loc=0)

plt.subplot(122)
plt.plot(x,y1)
plt.plot(x,y2)
plt.plot(x, mu,'--', color='black', label = '$\mu$', alpha=0.6)
plt.fill_between(np.linspace(0.,20., 2), -2, 2, alpha = 0.3)
plt.legend(loc=1)
plt.yticks([])
plt.xticks([])
plt.show()

y1 = np.random.multivariate_normal(mu, K21)
y2 = np.random.multivariate_normal(mu, K22)
plt.figure(figsize = (12,4))
plt.subplot(121)
plt.plot(x_r, y_RQ1, label = '$l = 0.5, a = 0.5$')
plt.plot(x_r, y_RQ2, label = '$l = 1.5, a = 2.5$')
plt.xlim([-5,5])
plt.ylim([-0.01,0.3])
plt.yticks([0,0.25], labels=[0,'$\sigma^2$'])
plt.xlabel('$x - x^{\prime}$')
plt.ylabel('$\kappa(x,x^{\prime}) $')
plt.legend(loc=0, fontsize=10)

plt.subplot(122)
plt.plot(x,y1)
plt.plot(x,y2)
plt.plot(x, mu,'--', color='black', label = '$\mu$', alpha=0.6)
plt.fill_between(np.linspace(0.,20., 2), -2, 2, alpha = 0.3)
plt.legend(loc=1)
plt.yticks([])
plt.xticks([])
plt.show()

y1 = np.random.multivariate_normal(mu, K11*K21)
y2 = np.random.multivariate_normal(mu, K12*K22)
plt.figure(figsize = (12,4))
plt.subplot(121)
plt.plot(x_r, y_RQ1*y_Per1, label = '$l = 0.5, a = 0.5$')
plt.plot(x_r, y_RQ2*y_Per2, label = '$l = 1.5, a = 2.5$')
plt.xlim([-5,5])
plt.ylim([-0.01,0.3])
plt.yticks([0,0.25], labels=[0,'$\sigma^2$'])
plt.xlabel('$x - x^{\prime}$')
plt.ylabel('$\kappa(x,x^{\prime}) $')
plt.legend(loc=0, fontsize=10)

plt.subplot(122)
plt.plot(x,y1)
plt.plot(x,y2)
plt.plot(x, mu,'--', color='black', label = '$\mu$', alpha=0.6)
plt.fill_between(np.linspace(0.,20., 2), -2, 2, alpha = 0.3)
plt.legend(loc=1)
plt.yticks([])
plt.xticks([])
plt.show()
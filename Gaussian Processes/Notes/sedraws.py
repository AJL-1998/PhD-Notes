import numpy as np
import matplotlib.pyplot as plt

import numpy as np

def kernel(X1, X2, l=1., sigma_f=1.):
    sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
    return sigma_f**2 * np.exp(-0.5 * l**(-2) * sqdist)




plt.figure(figsize=(12, 8))
ax = plt.subplot(121)

plt.ylim([-0.1,1.1])
plt.xlim([-3,3])
plt.xticks([-2,-1,0,1,2])
plt.yticks([0, 1])

x = np.linspace(-3.,3.,100)
l = 0.1
y = np.exp(-0.5 * l**(-2)*x ** 2)
plt.plot(x, y, label = '$l = 0.1$')
l = 1.
y = np.exp(-0.5 * l**(-2)*x ** 2)
plt.plot(x, y, label = '$l = 1.0$')
l = 2.
y = np.exp(-0.5 * l**(-2)*x ** 2)
plt.plot(x, y, label='$l = 2.0$')
plt.xlabel('$x - x^{\prime}$')
plt.ylabel('$\kappa(x, x^{\prime})$')
plt.legend(loc=1)

plt.subplot(122)
x = np.vstack( np.linspace(0., 10., 500) )
mu = np.array([0 for i in range(500)])
plt.ylim([-3,3])
plt.xlim([0,10])
plt.xticks([])
plt.yticks([])
plt.fill_between(np.linspace(0.,10., 2), -2, 2, alpha = 0.3)
plt.plot(x, [0 for i in x], '--',color = 'black', alpha = 0.8, label = 'Mean function')
K1 = kernel(x, x, l=0.10)
y1 = np.random.multivariate_normal(mu, K1)
K2 = kernel(x, x, l=1.)
y2 = np.random.multivariate_normal(mu, K2)
K3 = kernel(x, x, l=2.)
y3 = np.random.multivariate_normal(mu, K3)
plt.plot(x, y1)
plt.plot(x, y2)
plt.plot(x, y3)
plt.legend(loc=1)

plt.subplots_adjust(bottom=0.25, top=0.75)
plt.savefig('sedraws.pdf')
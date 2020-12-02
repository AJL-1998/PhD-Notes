import numpy as np
import matplotlib.pyplot as plt

import numpy as np

def kernel(X1, X2, l=1., sigma_f=1.0):
    """
    Isotropic squared exponential kernel.
    
    Args:
        X1: Array of m points (m x d).
        X2: Array of n points (n x d).

    Returns:
        (m x n) matrix.
    """
    sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
    return sigma_f**2 * np.exp(-0.5 / l**2 * sqdist)

x  = np.vstack(np.linspace(-12,12, 11))
y = np.sin(x)
x_s =  np.vstack(np.linspace(-12., 12., 500))

K = kernel(x, x)
K_s = kernel(x,x_s)
K_ss = kernel(x_s, x_s)
K_i = np.linalg.inv(K)

# Equation (7)
mu_s = K_s.T.dot(K_i).dot(y)

# Equation (8)
cov_s = K_ss - K_s.T.dot(K_i).dot(K_s)
N = np.random.multivariate_normal(mu_s.ravel(), cov_s)

x_true = np.linspace(-13,13,500)
y_true = np.sin(x_true)
plt.figure()
plt.plot(x_s, mu_s + cov_s / 2, color = 'blue')
plt.plot(x_s, mu_s - cov_s / 2, color = 'blue')
plt.plot(x_s, mu_s, color = 'black')
plt.plot(x_true, y_true, color = 'red', linestyle = 'dashed')
plt.scatter(x, y, marker = 'x', color ='red')
plt.savefig('first_gp.pdf')
import numpy as np
import matplotlib.pyplot as plt

import numpy as np

def kernel(X1, X2, l=1., sigma_f=1.):
    sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
    return sigma_f**2 * np.exp(-0.5 / l**2 * sqdist)

def kernel2(X1, X2, c = 0, d = 1):
    return (np.sum(X1, 1).reshape(-1, 1) + np.sum(X2, 1) + c) ** d

z = []
var = 0.
for i in range(10):
    z_s = np.random.uniform(-12.,12.)
    z.append(z_s)
    x   = np.vstack(z)
    y   = 0.1*x*np.sin(x)
    x_s =  np.vstack(np.linspace(-12., 12., 500))
    
    K    = kernel(x, x)
    n = len(K)
    K += var * np.eye(n,n)
    K_s  = kernel(x,x_s)
    K_ss = kernel(x_s, x_s)
    K_i  = np.linalg.inv(K)
    
    # Equation (7)
    mu_s = K_s.T.dot(K_i).dot(y)
    mu_s = np.hstack(mu_s).tolist()
    
    # Equation (8)
    cov_s = K_ss - K_s.T.dot(K_i).dot(K_s)
    N = np.random.multivariate_normal(mu_s, cov_s)
    sigma = cov_s.diagonal()
    cov_p = mu_s + 2*np.sqrt(sigma)
    cov_m = mu_s - 2*np.sqrt(sigma)
    
    x_true = np.linspace(-12,12,500)
    y_true = 0.1*x_true*np.sin(x_true)
    plt.figure()
    plt.fill(np.concatenate([x_s, x_s[::-1]]),
             np.concatenate([cov_p,
                            (cov_m)[::-1]]),alpha=.5, fc='b', ec='None', label='95% confidence interval')
    plt.plot(x_s, mu_s, color = 'black')
    plt.plot(x_true, y_true, color = 'red', linestyle = 'dashed')
    plt.scatter(x, y, marker = '.', color ='red')
    plt.plot(x_s, N, color = 'm', alpha = .3)
    plt.show()
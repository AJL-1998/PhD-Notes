import numpy as np
import matplotlib.pyplot as plt

"""
A simple linear regression class for 1-D output. Will be further extended
    for 2-D output (i.e. 3-D plot).
"""

class LinearRegression:
    """
    DESCRIPTION
    ------------
        A basic class to perform 1-D, multiple linear regression
    
    ATTRIBUTES
    ------------
        x_vals: values x_i of independent variable x
        y_vals: values of observations y = f(x) + epsilon
        x_star: values x_i^* of independent variable we wish to test
        phi:    basis function
        D:      number of parameters to determine
    
    METHODS
    ------------
        param_form: returns the values of parameters theta, uses Cholesky
                        decomposition
        prediction_model: returns the values y_star
    
    
    """
    def __init__(self, x_vals, y_vals, x_star, phi, D):
        self.x_vals = x_vals
        self.y_vals = y_vals
        self.N = len(x_vals)
        self.N_star = len(x_star)
        self.x_star = x_star
        self.phi = phi
        self.D = D
        self.X = phi(x_vals[0], self.D)
        for i in range(1,self.N):
            self.X = np.vstack((self.X,phi(x_vals[i], self.D)))
        self.A = np.copy(self.X)
        self.A = self.A.T @ self.A
        
    def param_form(self):

        def LowerInv(L):
            n = len(L)
            X = np.zeros((n,n))
            for i in range(n):
                X[i,i] = 1/L[i,i]
                for j in range(i+1,n):
                    X[j,i] = -(1 / L[j,j]) * sum(L[j,k] * X[k,i] for k in range(0,j))
            return X
        
        
        A = np.copy(self.A)
        L = np.linalg.cholesky(A)
        Linv = LowerInv(L)
        Ainv = Linv.T @ Linv
        self.theta = Ainv @ self.X.T @ np.vstack(self.y_vals)
        
        
    def prediction_model(self):
        self.X_star = self.phi(self.x_star[0], self.D)
        for i in range(1,self.N_star):
            self.X_star = np.vstack((self.X_star,self.phi(self.x_star[i], self.D)))
        self.y_star = self.X_star @ self.theta

def kernel(x, d):
    return np.array([x ** i for i in range(d)])

n = 4
var = 10000
x_vals = np.random.uniform(0.0,50.0,50)
y_vals = 2 + x_vals - 4*x_vals ** 2 + 0.1* x_vals ** 3
epsilon = np.sqrt(var)*np.random.normal(size = 50)
y_vals = y_vals + epsilon
x_star = np.linspace(0.0,50.0,500)
linreg = LinearRegression(x_vals, y_vals, x_star, kernel, n)
linreg.param_form()
linreg.prediction_model()

plt.figure()
plt.scatter(x_vals, y_vals, marker = '.')
plt.plot(x_star, linreg.y_star, color = 'red')
plt.show()

print(linreg.theta)
import numpy as np
import matplotlib.pyplot as plt

"""
A simple linear regression class for 1-D output. Will be further extended
    for 2-D output (i.e. 3-D plot) and improved calls for matrix inversion
        of positive-definite matrices (Cholesky decomposition method).

A.J.Lee
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
        param_form: returns the values of parameters theta
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
        self.X = phi(x_vals[0])
        for i in range(1,self.N):
            self.X = np.vstack((self.X,phi(x_vals[i])))
        '''self.H = np.linalg.inv( np.copy(self.X).T @ np.copy(self.X) )
        self.theta = self.H @ self.X.T @ np.vstack(self.y_vals)'''
        
    def param_form(self):
        def Chol(A):
            n = len(A)
            L = np.zeros((n,n))
            for k in range(n):
                L[k,k] = np.sqrt(A[k,k] - sum(L[k,j] ** 2 for j in range(k)))
                for i in range(k,n):
                    L[i,k] = (1 / L[k,k]) * (A[i,k] - sum(L[i,j]*L[k,j] for j in range(i)))
            return L

        def LowerInv(L):
            n = len(L)
            X = np.zeros((n,n))
            for i in range(n):
                X[i,i] = 1/L[i,i]
                for j in range(i+1,n):
                    X[j,i] = -(1 / L[j,j]) * sum(L[j,k] * X[k,i] for k in range(0,j))
            return X
        
        A = np.copy(self.X)
        A = A.T @ A
        L = Chol(A)
        Linv = LowerInv(L)
        Ainv = Linv.T @ Linv
        self.theta = Ainv @ self.X.T @ np.vstack(self.y_vals)
        
        
    def prediction_model(self):
        self.X_star = self.phi(self.x_star[0])
        for i in range(1,self.N_star):
            self.X_star = np.vstack((self.X_star,self.phi(self.x_star[i])))
        self.y_star = self.X_star @ self.theta

def kernel(x):
    return np.array([x ** i for i in range(6)])

var = 10000
x_vals = np.random.uniform(0.0,100.0,200)
y_vals = 2 + x_vals + x_vals ** 2 - 0.01*x_vals ** 3
epsilon = np.sqrt(var)*np.random.normal(size = 200)
y_vals = y_vals + epsilon
x_star = np.linspace(0.0,100.0,500)
linreg = LinearRegression(x_vals, y_vals, x_star, kernel, 6)
linreg.param_form()
linreg.prediction_model()

plt.figure()
plt.scatter(x_vals, y_vals, marker = '.')
plt.plot(x_star, linreg.y_star, color = 'red')
plt.show
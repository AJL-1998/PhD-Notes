import numpy as np
import matplotlib.pyplot as plt

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
        self.H = np.linalg.inv( np.copy(self.X).T @ np.copy(self.X) )
        self.theta = self.H @ self.X.T @ np.vstack(self.y_vals)
        
    def prediction_model(self):
        self.X_star = self.phi(self.x_star[0])
        for i in range(1,self.N_star):
            self.X_star = np.vstack((self.X_star,self.phi(self.x_star[i])))
        self.y_star = self.X_star @ self.theta

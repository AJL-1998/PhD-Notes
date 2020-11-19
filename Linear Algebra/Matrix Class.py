import numpy as np

class nummatrix:
    """
    DESCRIPTION
    ------------
        A class for matrices using numpy.array
        
    ATTRIBUTES
    ------------
        nd    : dimensions of matrix i.e. [3,2] is a 3 x 2 matrix
        body  : a LIST of the matrix elements
                    read from left to right (MUST BE OF TYPE LIST AND NUMERICAL)
                        e.g.
                            {nd = [3,3], body = [i for i in range(9)]} 
                                represents
                                    np.array([[0,1,2],[3,4,5],[6,7,8]])
        shape : 'square', 'positive-definite', etc...
        
    METHODS
    ------------
        __inv__        : numpy's basic inverse method, only use
                                if shape is unknown square
        HouseholderAlg : Householder's method for reduction to Upper
                                Hessenberg form
    
    
    
    
    
    
    
    A.J. LEE
    
    """
    def __init__(self, nd = [0,0], body = [], shape = 'unknown'):
        if nd[0] == nd[1] and shape != 'square':
            shape += ' square'
        self.shape = shape
        self.A = np.zeros((nd[0],nd[1]))
        if len(body) < nd[0]*nd[1]:
            a = [0 for i in range(nd[0]*nd[1] - len(body))]
            body.extend(a)
        
        for i in range(nd[0]):
            for j in range(nd[1]):
                self.A[i,j] = body[j + nd[1]*i]
                
    def __inv__(self):
        C = np.linalg.inv(self.A)
        return C
                
    def HouseholderAlg(self):
        """
    

        Parameters
        ----------
        X : SQUARE MATRIX.

        Returns
        -------
        X : UPPER HESSENBERG FORM OF X.

        """
        if 'square' in self.shape:
            X = self.A
        else:
            return 'Matrix not square'
        

        def matmul(M, X, a, b):
            N = len(M)
            MX = np.zeros((N,N))
            for i in range(a,N):
                for j in range(b,N):
                    MX[i,j] = sum(M[i,k]*X[k,j] for k in range(N))
            return MX
        def rowret(X, a):
            n = len(X)
            A = np.zeros((n,n))
            A[0:a,:] = X[0:a,:]
            return A
        def colret(X, a):
            n = len(X)
            A = np.zeros((n,n))
            A[:,0:a] = X[:,0:a]
            return A
        n = len(X)
        HX = np.zeros((n,n))
        HXH = np.zeros((n,n))
        for k in range(n-2):
            A = np.zeros((n,n))
            v = np.zeros((n,1))
            e = np.zeros((n,1))
            e[k+1] = 1
        
            if X[k+1,k] < 0:
                SG = -1
            else:
                SG = 1
            for j in range(k+1,n):
                v[j] = X[j,k]
            sigma = np.sqrt(sum(v[i] ** 2 for i in range(n)))
            v = v + SG*sigma*e
            v = v / np.sqrt(sum(v[i] ** 2 for i in range(n)))
            for i in range(k+1, n):
                for j in range(k+1,n):
                    A[i,j] = v[i]*v[j]
            H = np.identity(n) - 2 * A
        
            HX = rowret(X,k) + matmul(H, X, k, 0)
            HXH = colret(HX,k)  + matmul(HX, H, 0, k)
        
            X = HXH
        
        for i in range(n):
            for j in range(n):
                if i > j + 1 :
                    X[i,j] = 0
        
        return X
    
x = nummatrix(nd = [5,5],body = np.random.rand(18).tolist())
print(x.A)
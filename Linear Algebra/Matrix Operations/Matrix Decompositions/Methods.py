import numpy as np
import math
"""

A set of methods for matrix decompositions. Currently includes
- PLU decomposition A = PLU
- Cholesky decomposition A = LL ** T
- Inverse of a lower triangular matrix
- Inverse of an upper triangular matrix

A.J.Lee

"""


def pivotize(m):
    """
    
    Creates the pivoting matrix for m with regards to PLUfac.
    
    """
    
    n = len(m)
    ID = [[float(i == j) for i in range(n)] for j in range(n)]
    for j in range(n):
        row = max(range(j, n), key=lambda i: abs(m[i][j]))
        if j != row:
            ID[j], ID[row] = ID[row], ID[j]
    return ID

def PLUfac(x):
    """
    
    Returns matrices P, L, U of the decomposition:
    
    Px = LU.
    
    """
    
    n = len(x)
    L = np.identity(n)
    U = x
    P = np.identity(n)
    p = pivotize(x)
    swap = [np.argmax(p[i]) for i in range(n)]
    for i in range(n):
        U[[i,swap[i]], :] = U[[swap[i],i],:]
        P[[i,swap[i]], :] = P[[swap[i],i],:]
    
    for i in range(n):
        if U[i,i] == 0 :
            continue
        for j in range(i+1,n):
            L[j,i] = U[j,i]/U[i,i]
            for k in range(i,n):
                U[j,k] = U[j, k] - L[j,i]*U[i,k]
    return P, L, U

def Chol(A):
    """
    

    Parameters
    ----------
    A : POSITIVE-DEFINITE MATRIX.

    Returns
    -------
    L : LOWER TRIANGULAR MATRIX SUCH THAT A = LL ** T.

    """
    
        
    n = len(A)
    L = np.zeros((n,n))
    for k in range(n):
        L[k,k] = math.sqrt(A[k,k] - sum(L[k,j] ** 2 for j in range(k)))
        for i in range(k,n):
            L[i,k] = (1 / L[k,k]) * (A[i,k] - sum(L[i,j]*L[k,j] for j in range(i)))
    
    return L

def LowerInv(L):
    """
    

    Parameters
    ----------
    L : LOWER TRIANGULAR MATRIX.

    Returns
    -------
    X : INVERSE MATRIX L ** -1.

    """
    n = len(L)
    X = np.zeros((n,n))
    for i in range(n):
        X[i,i] = 1/L[i,i]
        for j in range(i+1,n):
            X[j,i] = -(1 / L[j,j]) * sum(L[j,k] * X[k,i] for k in range(0,j))
    return X

def UpperInv(U):
    """
    

    Parameters
    ----------
    U : UPPER TRIANGULAR MATRIX.

    Returns
    -------
    X : INVERSE MATRIX U ** -1.

    """
    U = np.transpose(U)
    X = np.transpose(LowerInv(U))
    
    return X

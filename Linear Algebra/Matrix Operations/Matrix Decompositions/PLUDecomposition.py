import numpy as np

def pivotize(m):
    """
    
    Creates the pivoting matrix for m.
    
    """
    n = len(m)
    ID = [[float(i == j) for i in range(n)] for j in range(n)]
    for j in range(n):
        row = max(range(j, n), key=lambda i: abs(m[i][j]))
        if j != row:
            ID[j], ID[row] = ID[row], ID[j]
    return ID

def LUfac(x):
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

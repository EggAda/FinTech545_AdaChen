import numpy as np
import pandas as pd
from numpy.linalg import norm, eigh, svd, eigvals, inv, cholesky
import sys
import itertools

def near_psd_cov(a, epsilon=1e-10):
    n = a.shape[0]
    invSD = None
    out = a.copy()

    # Calculate the correlation matrix if we got a covariance matrix
    if not np.allclose(np.diag(out), 1.0):
        invSD = np.diag(1.0 / np.sqrt(np.diag(out)))
        out = invSD @ out @ invSD

    # SVD, update the eigenvalue and scale
    vals, vecs = eigh(out)
    vals = np.maximum(vals, epsilon)
    T = 1.0 / (vecs * vecs @ vals[:, np.newaxis])
    T = np.diag(np.sqrt(T.squeeze()))
    l = np.diag(np.sqrt(vals))
    B = T @ vecs @ l
    out = B @ B.T

    # Add back the variance
    if invSD is not None:
        invSD = np.diag(1.0 / np.diag(invSD))
        out = invSD @ out @ invSD

    return out


def near_psd_corr(a, epsilon=0):
    n = a.shape[0]
    out = np.array(a, copy = True)
    if not np.allclose(np.diag(out), 1):
        invSD = np.diag(1.0 / np.sqrt(np.diag(out)))
        out = np.dot(invSD, np.dot(out, invSD))
        
    vals, vecs = eigh(out)
    vals = np.maximum(vals, epsilon)
    
    T = np.diag(1.0 / np.sqrt(np.sum(vecs ** 2 * vals, axis = 1)))
    l = np.diag(np.sqrt(vals))
    B = np.dot(np.dot(T, vecs), l)
    out = np.dot(B, B.T)
    
    if 'invsD' in locals():
        invSD = np.diag(1.0 / np.diag(invSD))
        out = np.dot(invSD,np.dot(out, invSD))
        
    return out

def F_Norm(matrix):
    """Calculate the Frobenius norm of a matrix."""
    return np.sqrt(np.sum(np.square(matrix)))

def frobenius(input):
    result = 0
    for i in range(len(input)):
        for j in range(len(input)):
            result += input[i][j]**2
    return result

def higham_cov(input):
    weight = np.identity(len(input))
        
    norml = np.inf
    Yk = input.copy()
    Delta_S = np.zeros_like(Yk)
    
    invSD = None
    if np.count_nonzero(np.diag(Yk) == 1.0) != input.shape[0]:
        invSD = np.diag(1 / np.sqrt(np.diag(Yk)))
        Yk = invSD @ Yk @ invSD
    
    Y0 = Yk.copy()

    for i in range(1000):
        Rk = Yk - Delta_S
        Xk = np.sqrt(weight)@ Rk @np.sqrt(weight)
        vals, vecs = np.linalg.eigh(Xk)
        vals = np.where(vals > 0, vals, 0)
        Xk = np.sqrt(weight)@ vecs @ np.diagflat(vals) @ vecs.T @ np.sqrt(weight)
        Delta_S = Xk - Rk
        Yk = Xk.copy()
        np.fill_diagonal(Yk, 1)
        norm = frobenius(Yk-Y0)
        min_val = np.real(np.linalg.eigvals(Yk)).min()
        if abs(norm - norml) < 1e-8 and min_val > -1e-9:
            break
        else:
            norml = norm
    
    if invSD is not None:
        invSD = np.diag(1 / np.diag(invSD))
        Yk = invSD @ Yk @ invSD
    return Yk

def highams_corr(a, epsilon=1e-10):
    a = (a + a.T) / 2
    n = a.shape[0]
    t = np.eye(n) * epsilon
    a_new = a + t
    eigvals, eigvecs = eigh(a_new)

    # Adjust negative eigenvalues to epsilon
    eigvals = np.maximum(eigvals, epsilon)
    a_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T

    # Normalize the matrix to ensure it remains a correlation matrix
    diag_sqrt_inv = 1 / np.sqrt(np.diag(a_psd))
    a_psd = a_psd * diag_sqrt_inv[:, None] * diag_sqrt_inv[None, :]
    
    # Ensure the diagonal elements are exactly 1
    np.fill_diagonal(a_psd, 1)
    
    return a_psd

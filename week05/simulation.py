import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from scipy.linalg import cholesky, eigh
from numpy.linalg import LinAlgError
from nearpsd_higham import *
from chol import *

def simulate_normal(N, cov, mean=None, seed=1234, fix_method=chol_pd):
    np.random.seed(seed)
    n = cov.shape[0]

    if cov.shape[0] != cov.shape[1]:
        raise ValueError(f"Covariance matrix is not square: {cov.shape}")

    if mean is None:
        mean = np.zeros(n)
    elif len(mean) != n:
        raise ValueError(f"Mean vector length ({len(mean)}) does not match covariance matrix size ({n}).")

    eigenvalues, _ = np.linalg.eig(cov)
    if min(eigenvalues) < 0:
        cov = near_psd_cov(cov) if fix_method == 'near_psd' else fix_method(cov)
        
    L = chol_pd(cov)
    samples = np.dot(np.random.randn(N, n), L.T) + mean
    return samples.T


def simulate_normal_higham(N, cov, mean=None, seed=1234, fix_method=higham_cov):
    np.random.seed(seed)
    n = cov.shape[0]

    if cov.shape[0] != cov.shape[1]:
        raise ValueError(f"Covariance matrix is not square: {cov.shape}")

    if mean is None:
        mean = np.zeros(n)
    elif len(mean) != n:
        raise ValueError(f"Mean vector length ({len(mean)}) does not match covariance matrix size ({n}).")

    eigenvalues, _ = np.linalg.eig(cov)
    if min(eigenvalues) < 0:
        cov = higham_cov(cov) if fix_method == 'higham' else fix_method(cov)

    L = chol_pd(cov)
    samples = np.dot(np.random.randn(N, n), L.T) + mean
    return samples.T


def simulate_pca(N, cov_matrix, mean=None, seed=1234, pctExp=1):
    m, n = cov_matrix.shape
    if n != m:
        raise ValueError(f"Covariance Matrix is not square ({n},{m})")
    
    if mean is None:
        mean = np.zeros(n)
    else:
        if len(mean) != n:
            raise ValueError(f"Mean ({len(mean)}) is not the size of cov ({n},{n})")
    
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[indices]
    eigenvectors = eigenvectors[:, indices]
    
    tv = np.sum(eigenvalues)
    posv = np.where(eigenvalues > 1e-8)[0]
    nval = len(posv) 
    if pctExp < 1:
        pct = 0.0
        for i, val in enumerate(eigenvalues[posv]):
            pct += val / tv
            if pct >= pctExp:
                nval = i + 1  # Adjust nval based on the percentage explained
                break
    
    posv = posv[:nval]  # Truncate posv based on nval
    eigenvalues = eigenvalues[posv]
    eigenvectors = eigenvectors[:, posv]
    
    B = eigenvectors @ np.diag(np.sqrt(eigenvalues))
    np.random.seed(seed)
    rand_normals = np.random.normal(0.0, 1.0, size=(N, len(posv)))
    out = np.dot(rand_normals, B.T) + mean
    
    return out.T
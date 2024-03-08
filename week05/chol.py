import numpy as np

def chol_pd(cov):
    a = np.array(cov)
    n = a.shape[0]
    root = np.zeros_like(a)
    
    for j in range(n):
        # Calculate the sum of squares of elements up to the j-th column
        s = np.sum(root[j, :j] ** 2)
        
        # Compute the diagonal element and check for non-positive values
        try:
            root[j, j] = np.sqrt(max(a[j, j] - s, 0))
        except ValueError:
            raise np.linalg.LinAlgError("Matrix is not positive definite")

        for i in range(j + 1, n):
            # Use dot product for the off-diagonal elements
            if root[j, j] > 0:  # Avoid division by zero
                root[i, j] = (a[i, j] - np.dot(root[i, :j], root[j, :j])) / root[j, j]
            else:
                root[i, j] = 0
    return root

# # Cholesky factorization
# def chol_pd(a):
#     n = a.shape[0]
#     root = np.zeros((n, n))

#     for j in range(n):
#         s = np.sum(root[j, :j] ** 2)
#         try:
#             root[j, j] = np.sqrt(a[j, j] - s)
#         except ValueError:
#             raise np.linalg.LinAlgError("Matrix is not positive definite")
        
#         for i in range(j+1, n):
#             s = np.dot(root[i, :j], root[j, :j])
#             root[i, j] = (a[i, j] - s) / root[j, j]
    
#     return root
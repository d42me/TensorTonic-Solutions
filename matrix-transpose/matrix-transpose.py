import numpy as np

def matrix_transpose(A: list) -> np.ndarray:
    """
    Returns the transposed matrix as a NumPy array.
    """
    # Write code here
    a = np.asarray(A)
    n, m = a.shape
    T = np.zeros((m, n), dtype=a.dtype)

    for i in range(n):
        for j in range(m):
            T[j, i] = a[i, j]

    return T
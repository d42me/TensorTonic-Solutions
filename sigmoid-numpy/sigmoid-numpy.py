import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    # Write code here
    exp_x = np.array(x)
    return 1 / (1 + np.exp(-exp_x))
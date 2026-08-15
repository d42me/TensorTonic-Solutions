import torch

def initialize_weights(fan_in, fan_out, method):
    """
    Returns: tensor of shape (fan_out, fan_in) with initialized weights
    """
    w = torch.empty(fan_out, fan_in)

    if method == "xavier_uniform":
        limit = 6 / (fan_in + fan_out)
        w = w.uniform_(-limit**0.5, limit**0.5)
    elif method == "xavier_normal":
        w = w.normal_(0, (2/(fan_in+fan_out))**0.5)
    elif method == "he_uniform":
        x = (6/fan_in)**0.5
        w = w.uniform_(-x, x)
    elif method == "he_normal":
        w = w.normal_(0, (2/fan_in)**0.5)

    return w
import torch

def activate(x, method="relu"):
    """
    Returns: list (activated tensor converted via .tolist())
    """
    x = torch.tensor(x, dtype=torch.float32)

    if method == "relu":
        return torch.maximum(torch.zeros_like(x), x).tolist()
    elif method == "sigmoid":
        return (1 / (1 + torch.exp(-x))).tolist()
    elif method == "tanh":
        return ((torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))).tolist()
    elif method == "leaky_relu":
        return torch.where(x > 0, x, 0.01 * x).tolist()
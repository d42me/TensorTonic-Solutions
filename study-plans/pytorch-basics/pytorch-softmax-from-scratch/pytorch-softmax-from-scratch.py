import torch

def softmax(logits):
    """
    Returns: tensor of same shape with softmax probabilities (each row sums to 1)
    """
    max_vals = torch.max(logits, dim=1, keepdim=True).values
    shifted = logits - max_vals
    exps = torch.exp(shifted)
    return exps / exps.sum(dim=1, keepdim=True)

import torch

def compute_loss(pred, target, method, delta=1.0):
    """
    Returns: float, the mean loss value
    """
    pred = torch.tensor(pred, dtype=torch.float32)
    if method == "mse":
        target = torch.tensor(target, dtype=torch.float32)
        return ((pred - target)**2).mean().item()
    elif method == "cross_entropy":
        target = torch.tensor(target, dtype=torch.long)
        m = pred.amax(dim=1, keepdim=True)                # (N, 1)
        lse = m + (pred - m).exp().sum(dim=1, keepdim=True).log()    # (N, 1)
        target_logit = pred.gather(1, target.unsqueeze(1))      # (N, 1)
        return (lse - target_logit).squeeze(1).mean()  
    elif method == "huber":
        target = torch.tensor(target, dtype=torch.float32)
        diff = (pred - target).abs()
        loss = torch.where(diff > delta, delta * (diff - 0.5 * delta), 0.5 * diff**2)
        return loss.mean().item()
        
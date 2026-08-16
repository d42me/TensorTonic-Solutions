import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

def create_balanced_loader(features, labels, batch_size):
    """
    Returns: a DataLoader that oversamples underrepresented classes
    """
    num_of_samples = len(features)
    class_counts = torch.bincount(labels)
    weights = 1.0 / class_counts[labels]
    sampler = WeightedRandomSampler(weights, num_of_samples, replacement=True)
    return DataLoader(TensorDataset(features, labels), batch_size=batch_size, sampler=sampler)

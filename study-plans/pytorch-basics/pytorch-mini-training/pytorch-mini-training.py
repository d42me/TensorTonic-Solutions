import torch
import torch.nn as nn

def train_epoch(model, dataloader, criterion, optimizer):
    """
    Returns: average loss over all batches (float)
    """
    model.train()
    total_batches = 0
    acc_loss = 0.0

    for inputs, targets in dataloader:
        outputs = model(inputs)
        
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_batches += 1
        acc_loss += loss.item()

    return acc_loss / total_batches
        

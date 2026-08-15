import torch
import torch.nn as nn

def train_with_scheduler(model, dataloader, criterion, optimizer, scheduler, num_epochs):
    """
    Returns: dict with 'losses' (list of per-epoch avg loss) and 'lrs' (list of learning rate per epoch)
    """
    losses = []
    lrs = []
    for epoch in range(num_epochs):

        total_loss = 0.0
        total_lr = 0.0
        n_batches = 0
        
        model.train()

        for input, target in dataloader:
            optimizer.zero_grad()

            output = model(input)

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if n_batches == 0:
                total_lr = optimizer.param_groups[0]["lr"]

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        losses.append(total_loss / n_batches)
        lrs.append(total_lr)

    return {"losses": losses, "lrs": lrs}
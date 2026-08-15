import torch
import torch.nn as nn

def train_with_early_stopping(model, train_loader, val_loader, criterion, optimizer, max_epochs, patience):
    """
    Returns: dict with 'train_losses' (list), 'val_losses' (list), 'stopped_epoch' (int, 1-indexed)
    """
    train_losses = []
    val_losses = []
    consecutive_misses = 0
    
    for epoch_count in range(max_epochs):
        # Traning
        model.train()
        train_loss = 0.0
        n_train = 0

        if consecutive_misses >= patience:
            return {"train_losses": train_losses, "val_losses": val_losses, "stopped_epoch": epoch_count}
        
        for x, y in train_loader:
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_train += 1

        # Eval
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                output = model(x_val)
                loss_val = criterion(output, y_val)
                val_loss += loss_val.item()
                n_val += 1

        val_loss_avg = val_loss / n_val
        if len(val_losses) == 0 or val_loss_avg < min(val_losses):
            consecutive_misses = 0
        else:
            consecutive_misses += 1

        train_losses.append(train_loss / n_train)
        val_losses.append(val_loss_avg)

    return {"train_losses": train_losses, "val_losses": val_losses, "stopped_epoch": max_epochs}      
    

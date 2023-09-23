import torch
from torch.utils.data import DataLoader
from torch import nn
from typing import Tuple

def calculate_validation_loss(model: nn.Module, test_loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    total_loss = 0.0
    total_samples = 0

    model.eval()
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs.float())
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)  # Accumulate the total loss
            total_samples += inputs.size(0)  # Accumulate the total number of samples

    model.train()

    return total_loss / total_samples

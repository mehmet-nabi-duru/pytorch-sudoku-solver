from tqdm import tqdm
from CNN.utils.validation import calculate_validation_loss
from CNN.utils.loss_curve_plotter import LossCurvePlotter
from CNN.utils.early_stopper import EarlyStopper
from CNN.trainer import ITrainer
import torch.nn as nn
import torch
from typing import List, Tuple

class ModelTrainer(ITrainer):
    """
    ModelTrainer Class implementation
    """

    def __init__(self, early_stopper: EarlyStopper, loss_curve_plotter: LossCurvePlotter) -> None:
        self.early_stopper = early_stopper
        self.loss_curve_plotter = loss_curve_plotter

    def train_model(self, net: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    criterion: nn.Module,
                    train_loader: torch.utils.data.DataLoader,
                    test_loader: torch.utils.data.DataLoader,
                    device: torch.device,
                    num_epochs: int = 50) -> Tuple[List[float], List[float], List[float], List[float]]:
        
        net.to(device)
        epoch_log = []
        train_loss_log = []
        validation_loss_log = []
        accuracy_log = []

        for epoch in tqdm(range(num_epochs)):
            net.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                outputs = net(inputs.float())
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                # Calculate accuracy for this mini-batch
                _, predicted = torch.max(outputs.data, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            # Calculate the training accuracy for this epoch
            accuracy = 100 * correct / total
            accuracy_log.append(accuracy)

            # Calculate the training loss for this epoch
            train_loss = running_loss / len(train_loader)
            train_loss_log.append(train_loss)
            epoch_log.append(epoch + 1)

            # Calculate the validation loss after each epoch
            validation_loss = calculate_validation_loss(net, test_loader, criterion, device)
            validation_loss_log.append(validation_loss)

            if self.early_stopper.early_stop(validation_loss):
                print("Early stopping triggered!")
                break

            # Print epoch-level information
            print(f'Epoch: {epoch + 1}, Loss: {train_loss:.3f}, Validation Loss: {validation_loss:.3f}, Test Accuracy: {accuracy:.3f}%')

            self.loss_curve_plotter.update_loss_logs(epoch + 1, train_loss, validation_loss)

        print('Training Completed.')
        self.loss_curve_plotter.plot_loss_curves()
        # Save the model
        model_script = torch.jit.script(net)
        torch.jit.save(model_script, "CNN\model.pt")
        return epoch_log, train_loss_log, validation_loss_log, accuracy_log
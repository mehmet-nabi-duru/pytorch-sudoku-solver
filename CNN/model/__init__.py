from abc import ABC, abstractmethod
import torch
import torch.nn as nn

class IModel(ABC):
    """
    Model interface
    """

    @abstractmethod
    def load_model_and_functions(*args, **kwargs):
        pass

    @abstractmethod
    def load_model_from_file(*args, **kwargs):
        pass


class Net(nn.Module):
    def __init__(self, num_classes=10) -> None:
        super(Net, self).__init__()


        # First convolutional layer: 1 input, 64 output channels
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), padding=1)  
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))  

        # Second convolutional layer: 64 input, 128 output channels
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))

        # Third convolutional layer: 128 input, 256 output channels
        self.conv3 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1)  
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2))  

        # Calculate the number of input features for the first fully connected layer
        self.fc_input_size = 256 * (28 // (2 ** 3)) * (28 // (2 ** 3))
        self.fc1 = nn.Linear(self.fc_input_size, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.relu4 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(256, 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.relu5 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(128, 64)
        self.bn6 = nn.BatchNorm1d(64)
        self.relu6 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.2)

        self.fc4 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)

        x = x.view(-1, self.fc_input_size)  # Flatten the tensor based on calculated size
        x = self.fc1(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn5(x)
        x = self.relu5(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = self.bn6(x)
        x = self.relu6(x)
        x = self.dropout3(x)

        x = self.fc4(x)

        return x

from typing import Tuple
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
import torch
from CNN.dataloader import IDataLoader
import pandas as pd
from sklearn.model_selection import train_test_split

class CSVDataLoader(IDataLoader):
    """
    DataLoader for this project
    """

    @staticmethod
    def load_data(dataset_path="CNN\dataset.csv", test_size:float = 0.2, *args, **kwargs) \
        -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Loads and preprocess the dataset
        Args:
            dataset_path (str, optional): Path to the dataset csv file. Default: "CNN\dataset.csv"
            test_size (float, optional): Fraction of the data to be used as the test set. Default: 0.2.

        Returns:
            X_train, X_test, y_train, y_test (Tuple[Tensor, Tensor, Tensor, Tensor]): A tuple containing the training data, test data, training labels and test labels in PyTorch Tensors

        Raises:
            FileNotFoundError: If the dataset file is not found.
            ValueError: If dataset is in invalid format or any missing columns
        """
        try:
            data = pd.read_csv(dataset_path, index_col=0)
        except FileNotFoundError as e:
            raise FileNotFoundError("Dataset file not found: {}".format(dataset_path)) from e
        
        # Check if the dataset contains required columns
        if "labels" not in data.columns:
            raise ValueError("Dataset file must contain 'labels' column.")
        if len(data.columns) != 785:
            raise ValueError("Invalid dataset format.")

        X = data.iloc[:,1:]
        X = X.values / 255
        X = X.reshape(X.shape[0], 1, 28, 28) # (n_sample, n_channel, n_pixel_width, n_pixel_height)
        y = data[["labels"]].values.reshape((-1, ))

        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        
        X_train, X_test, y_train, y_test = torch.from_numpy(X_train), torch.from_numpy(X_test), torch.from_numpy(y_train), torch.from_numpy(y_test)

        return X_train, X_test, y_train, y_test

    @staticmethod
    def create_data_loaders(X_train:Tensor , X_test: Tensor, 
                            y_train: Tensor, y_test: Tensor, 
                            batch_size: int = 128, *args, **kwargs) -> Tuple[DataLoader, DataLoader]:
        
        """
        Creates training and testing loaders

        Args:
            X_train (torch.Tensor): Training data features as a PyTorch Tensor.
            X_test (torch.Tensor): Test data features as a PyTorch Tensor.
            y_train (torch.Tensor): Training data labels as a PyTorch Tensor.
            y_test (torch.Tensor): Test data labels as a PyTorch Tensor.
            batch_size (int, optional): Batch size for the data loaders. Default: 128.

        Returns:
            train_loader, test_loader (Tuple[DataLoader, DataLoader]): A tuple containing train and test loaders
        """

        # Create a training dataset from X_train and y_train
        train = TensorDataset(X_train, y_train)
        # Create a testing dataset from X_test and y_test
        test = TensorDataset(X_test, y_test)

        # Create a training data loader from the train dataset
        # with a batch size of 128 and shuffling the data
        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)

        # Create a testing data loader from the test dataset
        # with a batch size of 128 and shuffling the data
        test_loader = DataLoader(test, batch_size=batch_size, shuffle=True)

        return train_loader, test_loader
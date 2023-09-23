from tqdm import tqdm
import numpy as np
from PIL import Image
import os
from typing import Tuple
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
import torch
from CNN.dataloader import IDataLoader
import pandas as pd
from sklearn.model_selection import train_test_split

class ImageDataLoader(IDataLoader):
    """
    Image Data Loader
    use this if dataset is in images format
    """
    @staticmethod
    def load_data(dataset_folder_path:str, test_size:float = 0.2, *args, **kwargs) \
        -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Loads and preprocess the data from the image folder

        Args:
            dataset_folder_path (str, optional): Path to the dataset csv file. Default: "CNN\dataset.csv"
            test_size (float, optional): Fraction of the data to be used as the test set. Default: 0.2.

        Returns:
            X_train, X_test, y_train, y_test (Tuple[Tensor, Tensor, Tensor, Tensor]): A tuple containing the training data, test data, training labels and test labels in PyTorch Tensors
        Raises:
            F
        """

        # Reading images

        # image dataset structured like this:
        # "label-n_th_example.png"

        # Get the total number of images in the folder
        files = [filename for filename in os.listdir(dataset_folder_path) if filename.endswith(".png")]

        # Define the arrays for pixel data and labels
        images = np.empty((len(files), 1, 28, 28), dtype=np.float32)
        labels = np.empty(len(files), dtype=np.int32)

        for i, image in tqdm(enumerate(files), total=len(files)):
            image_path = os.path.join(dataset_folder_path, image)
            with Image.open(image_path) as img:
                img = img.convert("L")
                img = img.resize((28,28))
                img = np.asarray(img) / 255 # convert it into np array and normalize
                label = int(image.split("-")[0])

            images[i, 0, :, :] = img
            labels[i] = label

        X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=test_size)
        
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
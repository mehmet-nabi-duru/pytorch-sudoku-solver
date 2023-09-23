import os
import torch
from CNN.model import IModel, Net
from CNN.utils.early_stopper import EarlyStopper
from CNN.trainer.trainer import ModelTrainer
from CNN.dataloader import IDataLoader
from CNN.utils.loss_curve_plotter import LossCurvePlotter
from typing import Type

class Model(IModel):
    """
    Model implementation
    """

    @staticmethod
    def load_model_and_functions(model=Net(), 
                                 criterion=torch.nn.CrossEntropyLoss(), 
                                 optimizer=torch.optim.Adam, 
                                 learning_rate=0.001, 
                                 *args, **kwargs):
        """
        Defines the model, criterion and optimizer
        """
        

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = model.to(device).float()
        criterion = criterion
        optimizer = optimizer(net.parameters(), lr=learning_rate)
        return net, criterion, optimizer
    
    @staticmethod
    def load_model_from_file(model_path="CNN\model\model.pt",
                             criterion=torch.nn.CrossEntropyLoss(), 
                             optimizer=torch.optim.Adam, 
                             learning_rate=0.001, 
                             *args, **kwargs):
        """
        Loads model from the script file
        """

        if os.path.exists(model_path):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            net = torch.jit.load(model_path, map_location=device)
            criterion = criterion
            optimizer = optimizer(net.parameters(), lr=learning_rate)
            return net#, criterion, optimizer

        else:
            # return Model.load_model_and_functions()
            raise FileNotFoundError(f"Model script file not found in \"{model_path}\". use Model.load_model_from_file() and train the model.")
        

 
class ModelFactory:
    def __init__(self, model_cls: Type[IModel] = Model):
        self.model_cls = model_cls

    def load_model(self, *args, **kwargs):
        model = self.model_cls.load_model_from_file(*args, **kwargs)
        return model

    def train_model(self, data_loader_type: Type[IDataLoader], dataset_path, model_type: Type[IModel] = Model, test_size=0.2, batch_size=128):
        """
        Usage:
            if dataset is in csv
                train_model(CSVDataLoader, "mydata.csv", Model)
            if dataset is in images
                train_model(ImageDataLoader, "myimages", Model)
        """

        # Create the data loader
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data_loader_factory = data_loader_type
        data_loader = data_loader_factory.create_data_loader(dataset_path, test_size=test_size)

        # Create the data loaders
        X_train, X_test, y_train, y_test = data_loader.load_data()
        train_loader, test_loader = data_loader.create_data_loaders(X_train, X_test, y_train, y_test, batch_size=batch_size)

        # Create the model
        model_factory = ModelFactory(model_type)
        net, criterion, optimizer = model_factory.create_model()

        # Create the early stopper and loss curve plotter
        early_stopper = EarlyStopper(patience=1, min_delta=0)
        loss_curve_plotter = LossCurvePlotter()

        # Train the model
        trainer = ModelTrainer(early_stopper, loss_curve_plotter)
        trainer.train_model(net, optimizer, criterion, train_loader, test_loader, device)


from CNN.dataloader.csv_loader import CSVDataLoader
from CNN.model.model import Model, ModelFactory

def main():

    data_loader = CSVDataLoader()
    X_train, X_test, y_train, y_test = data_loader.load_data(dataset_path="CNN\data\dataset.csv")
    train_loader, test_loader = data_loader.create_data_loaders(X_train, X_test, y_train, y_test)

    model_factory = ModelFactory()

    net = model_factory.load_model(model_path="CNN\model\model.pt")
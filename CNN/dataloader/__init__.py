# CNN/dataloader/__init__.py

from abc import ABC, abstractmethod
from torch import Tensor
from typing import Tuple
from torch.utils.data import DataLoader


class IDataLoader(ABC):
    """
    Data Loader Interface
    """

    @abstractmethod
    def load_data(*args, **kwargs) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        pass
    
    @abstractmethod
    def create_data_loaders(*args, **kwargs) -> Tuple[DataLoader, DataLoader]:
        pass
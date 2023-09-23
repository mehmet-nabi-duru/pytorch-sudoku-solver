from abc import ABC, abstractmethod
from typing import Tuple, List


class ITrainer(ABC):
    """Trainer interface"""

    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def train_model(self, *args, **kwargs) -> Tuple[List[float], List[float], List[float], List[float]]:
        pass

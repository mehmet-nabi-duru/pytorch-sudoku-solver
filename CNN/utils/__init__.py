from abc import ABC, abstractmethod

class IEarlyStopper(ABC):
    """Early Stopper interface"""

    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass
    
    @abstractmethod
    def early_stop(self, *args, **kwargs):
        pass
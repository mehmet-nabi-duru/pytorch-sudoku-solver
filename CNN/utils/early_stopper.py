from CNN.utils import IEarlyStopper


class EarlyStopper(IEarlyStopper):
    """My EarlyStopper implementation"""

    def __init__(self, patience=1, min_delta=0, *args, **kwargs):
        """
        Initialize EarlyStopper instance.

        Args:
            patience (int, optional): Number of epochs to wait for improvement in validation loss. Defaults to 1.
            min_delta (float, optional): Minimum change in validation loss that counts as improvement. Defaults to 0.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')  # Initialize minimum validation loss to positive infinity

    def early_stop(self, validation_loss) -> bool:
        """
        Check if early stopping condition is met.

        Args:
            validation_loss (float): Current validation loss.

        Returns:
            bool: True if early stopping condition is met, False otherwise.
        """
        if validation_loss < self.min_validation_loss - self.min_delta:
            # If validation loss has decreased by at least min_delta
            self.min_validation_loss = validation_loss
            self.counter = 0
        else:
            # If validation loss has not decreased by min_delta or increased
            self.counter += 1
            if self.counter >= self.patience:
                # If patience is exceeded, return True to trigger early stopping
                return True
        return False
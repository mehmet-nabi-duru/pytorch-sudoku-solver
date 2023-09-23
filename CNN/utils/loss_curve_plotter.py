import matplotlib.pyplot as plt

class LossCurvePlotter:
    def __init__(self):
        self.epoch_log = []
        self.train_loss_log = []
        self.validation_loss_log = []

    def update_loss_logs(self, epoch, train_loss, validation_loss):
        self.epoch_log.append(epoch)
        self.train_loss_log.append(train_loss)
        self.validation_loss_log.append(validation_loss)

    def plot_loss_curves(self):
        # Plotting the training and validation loss curves
        plt.figure(figsize=(16, 6))
        plt.plot(self.epoch_log, self.train_loss_log, label='Training Loss')
        plt.plot(self.epoch_log, self.validation_loss_log, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Curves')
        plt.legend()
        plt.grid(True)
        plt.show()
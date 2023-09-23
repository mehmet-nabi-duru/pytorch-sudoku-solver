import cv2
import numpy as np
import torch
from CNN.model.model import ModelFactory

class DigitRecognizer:
    def __init__(self, model_path="CNN\model\model.pt"):
        factory = ModelFactory()
        model = factory.load_model(model_path=model_path)
        self.model = model

    def set_model(self, model):
        self.model = model

    def recognize_digits(self, processed_cells):
        non_zero_elements = [(i, cell/255.0) for i, cell in enumerate(processed_cells) if type(cell) != int]
        non_zero_indices, non_zero_cells = zip(*non_zero_elements)

        non_zero_cells = np.array(non_zero_cells)
        non_zero_cells_tensor = torch.Tensor(non_zero_cells).unsqueeze(1)
        with torch.no_grad():
            output = self.model(non_zero_cells_tensor)
            _, preds = torch.max(output,1)
            preds = preds.tolist()

        preds_with_zeros = [0] * len(processed_cells)
        for index, pred in zip(non_zero_indices, preds):
            preds_with_zeros[index] = pred

        return preds_with_zeros
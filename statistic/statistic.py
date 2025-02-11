from torch import Tensor


class Statistic:
    def __init__(
            self,
            y_true: Tensor,
            y_pred: Tensor,
            loss: int | float | bool = None):
        # Calculating precision, recall, and F1 score using PyTorch
        tp = ((y_pred == 1) & (y_true == 1)).sum().item()
        fp = ((y_pred == 1) & (y_true == 0)).sum().item()
        fn = ((y_pred == 0) & (y_true == 1)).sum().item()

        self.precision = tp / (tp + fp) if tp + fp > 0 else 0
        self.recall = tp / (tp + fn) if tp + fn > 0 else 0
        self.f1 = 2 * (self.precision * self.recall) / (self.precision + \
                       self.recall) if (self.precision + self.recall) > 0 else 0
        self.loss = loss

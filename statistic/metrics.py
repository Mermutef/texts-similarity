from torch import Tensor


class Metrics:
    def __init__(
            self,
            y_true: Tensor,
            y_pred: Tensor,
            loss: int | float | bool = None,
    ):
        # Calculating precision, recall, and F1 score using PyTorch
        tp = ((y_pred == 1) & (y_true == 1)).sum().item() + 1e-16
        fp = ((y_pred == 1) & (y_true == 0)).sum().item() + 1e-16
        fn = ((y_pred == 0) & (y_true == 1)).sum().item() + 1e-16

        self.precision = tp / (tp + fp)
        self.recall = tp / (tp + fn)
        self.f1 = 2 * (self.precision * self.recall) / \
            (self.precision + self.recall)
        self.loss = loss

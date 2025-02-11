import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from statistic.statistic import Statistic


class ModelTemplate(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.train_stat: list[Statistic] = list()
        self.test_stat: Statistic | None = None

    def start_train(self, trainset: DataLoader, num_epochs: int) -> None:
        pass

    def do_test(self, testset: DataLoader) -> None:
        pass

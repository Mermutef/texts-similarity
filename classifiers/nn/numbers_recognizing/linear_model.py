import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
import torch.optim as optim

from statistic.statistic import Statistic


class NumbersRecognizer(nn.Module):
    def __init__(self):
        super().__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_stat: list[Statistic] = list()
        self.test_stat: Statistic | None = None

        self.fc1 = nn.Linear(784, 86)
        self.fc2 = nn.Linear(86, 86)
        self.fc3 = nn.Linear(86, 86)
        self.fc4 = nn.Linear(86, 10)

    def forward(self, x) -> Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)

    def start_train(self, trainset: DataLoader, num_epochs: int = 3) -> None:
        # Loss and optimizer
        criterion = F.nll_loss
        optimizer = optim.Adam(self.parameters(), lr=0.001)

        for epoch in range(num_epochs):
            y_true = []
            y_pred = []
            loss: Tensor | None = None
            for i, (X_train, y_train) in enumerate(trainset):
                X_train = X_train.to(self.device)
                y_train = y_train.to(self.device)

                # Forward pass
                outputs: Tensor = self(X_train.view(-1, 28 * 28))
                _, predicted = torch.max(outputs, 1)
                loss = criterion(outputs, y_train)
                y_pred.extend(predicted.cpu().numpy())
                y_true.extend(y_train.cpu().numpy())

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            self.train_stat.append(Statistic(
                y_true=torch.tensor(y_true),
                y_pred=torch.tensor(y_pred),
                loss=loss.item()
            ))
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

    def do_test(self, testset: DataLoader) -> None:
        self.eval()
        y_true = []
        y_pred = []

        for X_test, y_test in testset:
            X_test = X_test.to(self.device)
            outputs = self(X_test.view(-1, 28 * 28))
            _, predicted = torch.max(outputs, 1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(y_test.cpu().numpy())

        # Convert lists to tensors for calculation
        self.test_stat = Statistic(
            y_true=torch.tensor(y_true),
            y_pred=torch.tensor(y_pred),
        )

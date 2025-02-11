import torch
import torch.nn as nn
from torch import Tensor, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F


from losses.contrastive_loss import ContrastiveLoss
from statistic.statistic import Statistic


# create the Siamese Neural Network
class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.train_stat: list[Statistic] = list()
        self.test_stat: Statistic | None = None

        # Setting up the Sequential of CNN Layers
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(256, 384, kernel_size=3, stride=1),
            nn.ReLU(inplace=True)
        )

        # Setting up the Fully Connected Layers
        self.fc1 = nn.Sequential(
            nn.Linear(384, 1024),
            nn.ReLU(inplace=True),

            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 2)
        )

    def forward_once(self, x):
        # This function will be called for both images
        # Its output is used to determine the similiarity
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        # In this function we pass in both images and obtain both vectors
        # which are returned
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        return output1, output2

    def start_train(self, trainset: DataLoader, num_epochs: int = 100) -> None:
        criterion = ContrastiveLoss()
        optimizer = optim.Adam(self.parameters(), lr=0.001)

        # Iterate throught the epochs
        for epoch in range(num_epochs):
            y_true = []
            y_pred = []
            loss: Tensor | None = None
            # Iterate over batches
            for i, (X1_train, X2_train, y_train) in enumerate(trainset, 0):
                # Send the images and labels to device
                X1_train = X1_train.to(self.device)
                X2_train = X2_train.to(self.device)
                y_train = y_train.to(self.device)

                output1, output2 = self(X1_train, X2_train)
                predicted = self.predict(output1, output2)
                loss = criterion(output1, output2, y_train)
                y_pred.extend(predicted)
                y_true.extend(y_train.cpu().numpy())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            self.train_stat.append(Statistic(
                y_true=torch.tensor(y_true),
                y_pred=torch.tensor(y_pred),
                loss=loss.item()
            ))
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

    @staticmethod
    def predict(
            output1: Tensor,
            output2: Tensor,
            limit: int = 0.4) -> list[int]:
        res = []
        for o1, o2 in zip(output1, output2):
            res.append(1 if 1 /
                       (F.pairwise_distance(o1, o2).item() +
                        1) >= limit else 0)
        return res

    @staticmethod
    def test(output1: Tensor, output2: Tensor, limit: int = 0.4) -> int:
        return 1 if 1 / (F.pairwise_distance(output1,
                         output2).item() + 1) >= limit else 0

    def do_test(self, testset: DataLoader) -> None:
        self.eval()
        y_true = []
        y_pred = []

        for i, (X1_test, X2_test, y_test) in enumerate(testset, 0):
            # Send the images and labels to device
            X1_test = X1_test.to(self.device)
            X2_test = X2_test.to(self.device)
            y_test = y_test.to(self.device)

            output1, output2 = self(X1_test, X2_test)
            predicted = self.test(output1, output2)

            y_pred.append(predicted)
            y_true.extend(y_test.cpu().numpy())

        # Convert lists to tensors for calculation
        self.test_stat = Statistic(
            y_true=torch.tensor(y_true),
            y_pred=torch.tensor(y_pred),
        )

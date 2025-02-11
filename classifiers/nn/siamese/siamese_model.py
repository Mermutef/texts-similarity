import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from datasets.siamese_dataset import SiameseNetworkDataset
from losses.contrastive_loss import ContrastiveLoss

from random.random_settings import determine_random


# create the Siamese Neural Network
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

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


def main() -> None:
    determine_random()
    # Load the training dataset
    folder_dataset = datasets.ImageFolder(root="./data/faces/training/")

    # Resize the images and transform to tensors
    transformation = transforms.Compose([transforms.Resize((100, 100)),
                                         transforms.ToTensor()
                                         ])

    # Initialize the network
    siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset,
                                            transform=transformation)

    # Create a simple dataloader just for simple visualization
    vis_dataloader = DataLoader(siamese_dataset,
                                shuffle=True,
                                num_workers=2,
                                batch_size=8)

    # Extract one batch
    example_batch = next(iter(vis_dataloader))

    # Example batch is a list containing 2x8 images, indexes 0 and 1, an also the label
    # If the label is 1, it means that it is not the same person, label is 0,
    # same person in both images
    concatenated = torch.cat((example_batch[0], example_batch[1]), 0)

    # Load the training dataset
    train_dataloader = DataLoader(siamese_dataset,
                                  shuffle=True,
                                  num_workers=8,
                                  batch_size=64)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = SiameseNetwork().to(device)
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0005)

    counter = []
    loss_history = []
    iteration_number = 0

    # Iterate throught the epochs
    for epoch in range(100):

        # Iterate over batches
        for i, (img0, img1, label) in enumerate(train_dataloader, 0):

            # Send the images and labels to CUDA
            img0, img1, label = img0.to(device), img1.to(
                device), label.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Pass in the two images into the network and obtain two outputs
            output1, output2 = net(img0, img1)

            # Pass the outputs of the networks and label into the loss function
            loss_contrastive = criterion(output1, output2, label)

            # Calculate the backpropagation
            loss_contrastive.backward()

            # Optimize
            optimizer.step()

            # Every 10 batches print out the loss
            if i % 10 == 0:
                print(
                    f"Epoch number {epoch}\n Current loss {
                        loss_contrastive.item()}\n")
                iteration_number += 10

                counter.append(iteration_number)
                loss_history.append(loss_contrastive.item())

    # Locate the test dataset and load it into the SiameseNetworkDataset
    folder_dataset_test = datasets.ImageFolder(root="./data/faces/testing/")
    siamese_dataset = SiameseNetworkDataset(
        imageFolderDataset=folder_dataset_test,
        transform=transformation)
    test_dataloader = DataLoader(
        siamese_dataset,
        num_workers=2,
        batch_size=1,
        shuffle=True)

    # Grab one image that we are going to test
    dataiter = iter(test_dataloader)
    x0, _, _ = next(dataiter)

    for i in range(5):
        # Iterate over 5 images and test them with the first image (x0)
        _, x1, label2 = next(dataiter)

        # Concatenate the two images together
        concatenated = torch.cat((x0, x1), 0)

        output1, output2 = net(x0.to(device), x1.to(device))
        euclidean_distance = F.pairwise_distance(output1, output2)
        print(euclidean_distance.shape)
        print(euclidean_distance.norm())
        print(euclidean_distance.item() / euclidean_distance.norm().item())
        print(
            f'Dissimilarity: {
                euclidean_distance.item():.2f} - {
                euclidean_distance.item() /
                euclidean_distance.norm().item():.2f}')


if __name__ == '__main__':
    main()

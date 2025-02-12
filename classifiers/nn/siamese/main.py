import torch
import numpy as np
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, RandomSampler

from classifiers.nn.siamese.siamese_model import SiameseNetwork
from datasets.siamese_dataset import SiameseNetworkDataset
from determining.random_settings import determine_random


def main() -> None:
    determine_random()
    # Автоматическое определение, где запускать код: на Nvidia GPU или на CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Resize the images and transform to tensors
    transformation = transforms.Compose([transforms.Resize((100, 100)),
                                         transforms.ToTensor()
                                         ])

    train_dataset = SiameseNetworkDataset(
        image_folder_dataset=datasets.ImageFolder(
            root="../../../datasets/siamese/data/faces/training/"),
        transform=transformation)
    test_dataset = SiameseNetworkDataset(
        image_folder_dataset=datasets.ImageFolder(
            root="../../../datasets/siamese/data/faces/testing/"),
        transform=transformation)
    trainset = DataLoader(train_dataset,
                          sampler=RandomSampler(train_dataset),
                          num_workers=8,
                          batch_size=64)
    testset = DataLoader(
        test_dataset,
        num_workers=2,
        batch_size=1,
        sampler=RandomSampler(test_dataset))

    model = SiameseNetwork().to(device)
    model.start_train(trainset)
    model.do_test(testset)

    # Get the angles from 0 to 2 pie (360 degree) in narray object
    X = np.arange(0, len(model.train_stat), 1)

    # Using built-in trigonometric function we can directly plot
    # the given cosine wave for the given angles
    Y1 = [i.precision for i in model.train_stat]
    Y2 = [i.recall for i in model.train_stat]
    Y3 = [i.f1 for i in model.train_stat]
    Y4 = [i.loss for i in model.train_stat]

    # Initialise the subplot function using number of rows and columns
    figure, axis = plt.subplots(2, 2)

    # For Sine Function
    axis[0, 0].plot(X, Y1)
    axis[0, 0].set_title("Precision")

    # For Cosine Function
    axis[0, 1].plot(X, Y2)
    axis[0, 1].set_title("Recall")

    # For Tangent Function
    axis[1, 0].plot(X, Y3)
    axis[1, 0].set_title("F1")

    # For Tanh Function
    axis[1, 1].plot(X, Y4)
    axis[1, 1].set_title("Loss")

    # Combine all the operations and display
    plt.show()

    print()
    print(f'Precision: {model.test_stat.precision:.3f}')
    print(f'Recall: {model.test_stat.recall:.3f}')
    print(f'F1 Score: {model.test_stat.f1:.3f}')


if __name__ == '__main__':
    main()

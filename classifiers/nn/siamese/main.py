import torch

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

    for i, (stat) in enumerate(model.train_stat):
        print(i)
        print(f'Precision: {stat.precision:.3f}')
        print(f'Recall: {stat.recall:.3f}')
        print(f'F1 Score: {stat.f1:.3f}')
        print(f'Loss: {stat.loss:.3f}')

    print()
    print(f'Precision: {model.test_stat.precision:.3f}')
    print(f'Recall: {model.test_stat.recall:.3f}')
    print(f'F1 Score: {model.test_stat.f1:.3f}')


if __name__ == '__main__':
    main()

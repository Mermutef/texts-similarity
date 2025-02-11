import torch

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, RandomSampler

from classifiers.nn.numbers_recognizing.linear_model import NumbersRecognizer
from random.random_settings import determine_random


def main() -> None:
    determine_random()
    # Автоматическое определение, где запускать код: на Nvidia GPU или на CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Скачивание и загрузка набора данных MNIST
    transform = transforms.Compose([transforms.ToTensor()])
    train_data = datasets.MNIST(
        "data",
        train=True,
        download=False,
        transform=transform)
    test_data = datasets.MNIST(
        "data",
        train=False,
        download=False,
        transform=transform)

    # Преобразование полученного набора данных
    # RandomSampler перемешивает выборку, гарантируя, что каждый пример будет
    # взят ровно 1 раз
    train_sampler = RandomSampler(train_data)
    test_sampler = RandomSampler(test_data)
    trainset = DataLoader(
        dataset=train_data,
        batch_size=64,
        sampler=train_sampler)
    testset = DataLoader(
        dataset=test_data,
        batch_size=64,
        sampler=test_sampler)

    model = NumbersRecognizer().to(device)
    model.start_train(trainset)
    model.do_test(testset)

    for i, (stat) in enumerate(model.train_stat):
        print(i)
        print(f'Precision: {stat.precision}')
        print(f'Recall: {stat.recall}')
        print(f'F1 Score: {stat.f1}')
        print(f'Loss: {stat.loss}')

    print()
    print(f'Precision: {model.test_stat.precision}')
    print(f'Recall: {model.test_stat.recall}')
    print(f'F1 Score: {model.test_stat.f1}')


if __name__ == '__main__':
    main()

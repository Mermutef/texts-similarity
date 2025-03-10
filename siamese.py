#!/usr/bin/env python
# coding: utf-8
from transformers import GPT2Model, GPT2Tokenizer

from datasets.siamese.text_dataset import TextDataset

PROJECT_ROOT = "/home/mermutef/PycharmProjects/texts-similarity"

import sys

sys.path.insert(1, PROJECT_ROOT)

import argparse

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from torch import Tensor, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold
from sklearn.metrics import DistanceMetric

from datasets.siamese.siamese_dataset import SiameseNetworkDataset
from determining.random_settings import determine_random
from losses.contrastive_loss import ContrastiveLoss
from statistic.metrics import Metrics


def get_args():
    parser = argparse.ArgumentParser(
        description='PyTorch Siamese network Example')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        metavar='N',
        help='input batch size for training (default: 64)'
    )
    parser.add_argument(
        '--test-batch-size',
        type=int,
        default=1,
        metavar='N',
        help='input batch size for testing (default: 1)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        metavar='N',
        help='number of epochs to train (default: 100)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.01,
        metavar='LR',
        help='learning rate (default: 0.01)'
    )
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.001,
        metavar='M',
        help='Learning rate step gamma (default: 0.001)'
    )
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA training'
    )
    parser.add_argument(
        '--no-mps',
        action='store_true',
        default=False,
        help='disables macOS GPU training'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        default=False,
        help='quickly check a single pass'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=134,
        metavar='S',
        help='random seed (default: 1)'
    )
    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        metavar='N',
        help='how many batches to wait before logging training status'
    )
    parser.add_argument(
        '--save-model',
        action='store_true',
        default=False,
        help='For Saving the current Model'
    )
    parser.add_argument(
        '--n-splits',
        type=int,
        default=5,
        metavar='N',
        help='Number of splits for crossvalidation (default: 5)'
    )

    args = parser.parse_args([])
    return args


def test(device, test_loader: DataLoader) -> Metrics:
    gpt_model = GPT2Model.from_pretrained('openai-community/gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('openai-community/gpt2')
    gpt_model.to(device)

    gpt_model.eval()

    y_true = []
    y_pred = []

    with (torch.no_grad()):
        for (text1, text2, similarity) in test_loader:
            print(text1, text2, similarity)
            print("*" * 20)
            text1, text2, similarity = (
                tokenizer(text1, return_tensors='pt').to(device),
                tokenizer(text2, return_tensors='pt').to(device),
                similarity.to(device)
            )
            print(text1, text2, similarity)
            output1 = gpt_model(**text1)[0].mean(1)
            output2 = gpt_model(**text2)[0].mean(1)
            x1 = output1.detach().detach().cpu().numpy()
            x2 = output2.detach().detach().cpu().numpy()
            x1 = [x1[0] / np.linalg.norm(x1) if np.linalg.norm(x1) != 0 else x1[0]]
            x2 = [x2[0] / np.linalg.norm(x2) if np.linalg.norm(x2) != 0 else x2[0]]
            print(1 - DistanceMetric.get_metric('euclidean').pairwise(x1, x2)[0][0])
            pred = 1 if (1 - DistanceMetric.get_metric('euclidean').pairwise(x1, x2)[0][0]) >= 0.5 else 0
            y_pred.append(pred)
            y_true.extend(similarity.detach().cpu().numpy())

    return Metrics(y_true=torch.from_numpy(np.array(y_true)), y_pred=torch.from_numpy(np.array(y_pred)))


args = get_args()

use_cuda = not args.no_cuda and torch.cuda.is_available()
use_mps = not args.no_mps and torch.backends.mps.is_available()

determine_random(args.seed)

if use_cuda:
    device = torch.device("cuda")
elif use_mps:
    device = torch.device("mps")
else:
    device = torch.device("cpu")

train_kwargs = {'batch_size': args.batch_size}
test_kwargs = {'batch_size': args.test_batch_size}

if use_cuda:
    cuda_kwargs = {'num_workers': 1,
                   'pin_memory': True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

print(f"Running on {device}")

total_dataset = TextDataset(delimiter=',', encoding='windows-1251',
                            csv_file=f"{PROJECT_ROOT}/datasets/siamese/data/train.csv")

test_metrics = []
history = {}

for split_idx, (train_idx, test_idx) in enumerate(KFold(n_splits=args.n_splits, shuffle=True).split(total_dataset)):
    print(f"Running split #{split_idx + 1}")
    test_loader = DataLoader(
        total_dataset,
        sampler=SubsetRandomSampler(test_idx),
        **test_kwargs,
    )

    train_metrics = []
    test_metrics.append(test(device, test_loader))

for split_idx in history.keys():
    test_stat = test_metrics[split_idx]

    print(f"Split #{split_idx + 1}:")
    print(f'\tPrecision: {test_stat.precision:.3f}')
    print(f'\tRecall: {test_stat.recall:.3f}')
    print(f'\tF1 Score: {test_stat.f1:.3f}')

avg_precision = 0
avg_recall = 0
avg_f1 = 0
for metric in test_metrics:
    avg_precision += metric.precision
    avg_recall += metric.recall
    avg_f1 += metric.f1
print(f'AVG Precision: {avg_precision / len(test_metrics):.3f}')
print(f'AVG Recall: {avg_recall / len(test_metrics):.3f}')
print(f'AVG F1 Score: {avg_f1 / len(test_metrics):.3f}')

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import warnings

warnings.filterwarnings('ignore')

import logging

logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer
from extensions import notify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
import matplotlib.pyplot as plt
from tqdm import tqdm

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()


# Настройки воспроизводимости
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Триплетный датасет
class TripletDataset(Dataset):
    def __init__(self, data_dict, tokenizer, max_length=128):
        """
        data_dict: словарь {question_id: DataFrame}
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data_dict = data_dict
        self.question_ids = list(data_dict.keys())
        self.samples = self._prepare_samples()

    def _prepare_samples(self):
        samples = []
        for q_id, df in self.data_dict.items():
            # Для каждого ответа студента создаем отдельный сэмпл
            for _, row in df.iterrows():
                samples.append({
                    'question_id': q_id,
                    'answer': str(row['answer']),
                    'true_answer': str(row['true_answer']),
                    'mark': row['mark']
                })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        q_id = sample['question_id']
        anchor_text = sample['answer']
        positive_text = sample['true_answer']

        # Выбираем случайный другой вопрос
        other_qid = random.choice([q for q in self.question_ids if q != q_id])
        # Выбираем случайный ответ из другого вопроса
        other_df = self.data_dict[other_qid]
        negative_sample = other_df.sample(1).iloc[0]
        negative_text = str(negative_sample['answer'])

        # Токенизация
        anchor_enc = self.tokenizer(
            anchor_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        positive_enc = self.tokenizer(
            positive_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        negative_enc = self.tokenizer(
            negative_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'anchor_ids': anchor_enc['input_ids'].squeeze(0),
            'anchor_mask': anchor_enc['attention_mask'].squeeze(0),
            'positive_ids': positive_enc['input_ids'].squeeze(0),
            'positive_mask': positive_enc['attention_mask'].squeeze(0),
            'negative_ids': negative_enc['input_ids'].squeeze(0),
            'negative_mask': negative_enc['attention_mask'].squeeze(0)
        }


# Модель для эмбеддингов
class EmbeddingBERT(nn.Module):
    def __init__(self, model_name='bert-base-multilingual-cased'):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]  # [CLS] токен

    def normalize(self, embeddings):
        return F.normalize(embeddings, p=2, dim=1)


# Triplet Loss
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # Косинусные расстояния
        pos_sim = F.cosine_similarity(anchor, positive)
        neg_sim = F.cosine_similarity(anchor, negative)

        # Triplet loss
        losses = F.relu(neg_sim - pos_sim + self.margin)
        return losses.mean()


# Функции для обучения
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc="Training"):
        # Перемещаем данные на устройство
        anchor_ids = batch['anchor_ids'].to(device)
        anchor_mask = batch['anchor_mask'].to(device)
        positive_ids = batch['positive_ids'].to(device)
        positive_mask = batch['positive_mask'].to(device)
        negative_ids = batch['negative_ids'].to(device)
        negative_mask = batch['negative_mask'].to(device)

        # Получаем эмбеддинги
        anchor_emb = model(anchor_ids, anchor_mask)
        positive_emb = model(positive_ids, positive_mask)
        negative_emb = model(negative_ids, negative_mask)

        # Нормализуем эмбеддинги
        anchor_emb = model.normalize(anchor_emb)
        positive_emb = model.normalize(positive_emb)
        negative_emb = model.normalize(negative_emb)

        # Вычисляем потери
        optimizer.zero_grad()
        loss = criterion(anchor_emb, positive_emb, negative_emb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            anchor_ids = batch['anchor_ids'].to(device)
            anchor_mask = batch['anchor_mask'].to(device)
            positive_ids = batch['positive_ids'].to(device)
            positive_mask = batch['positive_mask'].to(device)
            negative_ids = batch['negative_ids'].to(device)
            negative_mask = batch['negative_mask'].to(device)

            anchor_emb = model(anchor_ids, anchor_mask)
            positive_emb = model(positive_ids, positive_mask)
            negative_emb = model(negative_ids, negative_mask)

            anchor_emb = model.normalize(anchor_emb)
            positive_emb = model.normalize(positive_emb)
            negative_emb = model.normalize(negative_emb)

            loss = criterion(anchor_emb, positive_emb, negative_emb)
            total_loss += loss.item()

    return total_loss / len(dataloader)


# Функция для вычисления метрик
def calculate_metrics(model, dataloader, device, threshold=0.7):
    model.eval()
    all_labels = []
    all_similarities = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            anchor_ids = batch['anchor_ids'].to(device)
            anchor_mask = batch['anchor_mask'].to(device)
            positive_ids = batch['positive_ids'].to(device)
            positive_mask = batch['positive_mask'].to(device)

            # Получаем эмбеддинги
            anchor_emb = model.normalize(model(anchor_ids, anchor_mask))
            positive_emb = model.normalize(model(positive_ids, positive_mask))

            # Вычисляем косинусное сходство
            similarities = F.cosine_similarity(anchor_emb, positive_emb)
            all_similarities.extend(similarities.cpu().numpy())

            # Метки: 1 если это настоящий положительный пример, 0 если негативный
            labels = [1] * len(similarities)  # Всегда положительная пара
            all_labels.extend(labels)

    # Конвертируем сходство в предсказания
    predictions = [1 if sim > threshold else 0 for sim in all_similarities]

    # Вычисляем метрики
    acc = accuracy_score(all_labels, predictions)
    f1 = f1_score(all_labels, predictions)
    precision = precision_score(all_labels, predictions)
    recall = recall_score(all_labels, predictions)
    auc = roc_auc_score(all_labels, all_similarities)

    return acc, auc, f1, precision, recall, all_similarities


# Основной пайплайн
def main():
    # Конфигурация
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 32
    EPOCHS = 50
    LR = 2e-5
    MARGIN = 0.5
    THRESHOLD = 0.7

    set_seed(42)

    # Загрузка данных
    dir_path = "corpus"
    question_files = {
        0: "question_0.csv",
        1: "question_1.csv",
        2: "question_2.csv",
        3: "question_3.csv",
        4: "question_4.csv",
        5: "question_5.csv",
        6: "question_6.csv",
        7: "question_7.csv"
    }

    data_dict = {}
    for q_id, file_name in question_files.items():
        df = pd.read_csv(os.path.join(dir_path, file_name))
        data_dict[q_id] = df

    # Разделение на train/test по вопросам
    all_questions = list(data_dict.keys())
    train_questions, val_questions = train_test_split(all_questions, test_size=0.2, random_state=42)

    train_dict = {q: data_dict[q] for q in train_questions}
    val_dict = {q: data_dict[q] for q in val_questions}

    # Токенизатор и даталоадеры
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

    train_dataset = TripletDataset(train_dict, tokenizer)
    val_dataset = TripletDataset(val_dict, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Модель и оптимизатор
    model = EmbeddingBERT().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = TripletLoss(margin=MARGIN)

    # Обучение
    best_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(EPOCHS):
        # # Динамическое увеличение маржи
        # if epoch > 10:
        #     criterion.margin = min(1.0, MARGIN + epoch * 0.05)

        # Обучение
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)

        # Оценка
        val_loss = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)

        # Вычисление метрик
        val_acc, val_auc, val_f1, val_precision, val_recall, _ = calculate_metrics(
            model, val_loader, device, THRESHOLD
        )

        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | Val AUC: {val_auc:.4f}")
        if epoch % 10 == 0:
            notify(f"Epoch {epoch + 1}/{EPOCHS}\n" +
                   f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}\n" +
                   f"Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | Val AUC: {val_auc:.4f}")

        # Сохранение лучшей модели
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'best_triplet_model.pth')

    # Сохранение кривых обучения
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Triplet Loss Training')
    plt.savefig('triplet_loss_curves.png')

    # Загрузка лучшей модели для финальной оценки
    model.load_state_dict(torch.load('best_triplet_model.pth'))
    test_acc, test_auc, test_f1, test_precision, test_recall, similarities = calculate_metrics(
        model, val_loader, device, THRESHOLD
    )

    print("\nFinal Evaluation:")
    print(f"Accuracy: {test_acc:.4f} | F1: {test_f1:.4f} | AUC: {test_auc:.4f}")
    print(f"Precision: {test_precision:.4f} | Recall: {test_recall:.4f}")
    notify("Final Evaluation:\n" +
           f"Accuracy: {test_acc:.4f} | F1: {test_f1:.4f} | AUC: {test_auc:.4f}\n" +
           f"Precision: {test_precision:.4f} | Recall: {test_recall:.4f}")

    # Визуализация распределения сходств
    plt.figure(figsize=(10, 6))
    plt.hist(similarities, bins=50, alpha=0.7)
    plt.axvline(x=THRESHOLD, color='r', linestyle='--', label=f'Threshold: {THRESHOLD}')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.title('Distribution of Cosine Similarities')
    plt.legend()
    plt.savefig('similarity_distribution.png')


# Функция предсказания
def predict_similarity(model, tokenizer, text1, text2, device='cpu'):
    model.eval()

    # Токенизация
    enc1 = tokenizer(
        text1,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    enc2 = tokenizer(
        text2,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids1 = enc1['input_ids'].to(device)
    attention_mask1 = enc1['attention_mask'].to(device)
    input_ids2 = enc2['input_ids'].to(device)
    attention_mask2 = enc2['attention_mask'].to(device)

    with torch.no_grad():
        emb1 = model(input_ids1, attention_mask1)
        emb2 = model(input_ids2, attention_mask2)

        # Нормализация и косинусное сходство
        emb1 = F.normalize(emb1, p=2, dim=1)
        emb2 = F.normalize(emb2, p=2, dim=1)
        similarity = F.cosine_similarity(emb1, emb2).item()

    return similarity


if __name__ == '__main__':
    try:
        notify("Эксперимент начался")
        main()
        notify("Эксперимент закончился")
    except Exception as e:
        notify(str(e))

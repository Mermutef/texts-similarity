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
from transformers import BertTokenizer, BertModel
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


# Триплетный датасет (упрощенный)
class TripletDataset(Dataset):
    def __init__(self, data_dict, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data_dict = data_dict
        self.question_ids = list(data_dict.keys())
        self.samples = self._prepare_samples()

    def _prepare_samples(self):
        samples = []
        for q_id, df in self.data_dict.items():
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


# Модель для эмбеддингов (упрощенная)
class EmbeddingBERT(nn.Module):
    def __init__(self, model_name='bert-base-multilingual-cased'):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        # Размораживаем все слои
        for param in self.bert.parameters():
            param.requires_grad = True
        # Умеренный dropout
        self.dropout = nn.Dropout(0.3)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state[:, 0, :]
        return self.dropout(embeddings)

    def normalize(self, embeddings):
        return F.normalize(embeddings, p=2, dim=1)


# Стандартный Triplet Loss
class TripletLoss(nn.Module):
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # Евклидовы расстояния
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)

        # Triplet loss
        losses = F.relu(pos_dist - neg_dist + self.margin)
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
        for batch in tqdm(dataloader, desc="Calculating metrics"):
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
            all_labels.extend([1] * len(similarities))  # Метка 1 для положительных пар

            # Добавляем негативные примеры
            negative_ids = batch['negative_ids'].to(device)
            negative_mask = batch['negative_mask'].to(device)
            negative_emb = model.normalize(model(negative_ids, negative_mask))

            neg_similarity = F.cosine_similarity(anchor_emb, negative_emb)
            all_similarities.extend(neg_similarity.cpu().numpy())
            all_labels.extend([0] * len(neg_similarity))  # Метка 0 для негативных пар

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
    BATCH_SIZE = 12
    EPOCHS = 50
    LR = 2e-5  # Стандартный LR для fine-tuning
    MARGIN = 0.5  # Умеренный margin
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

    # Обучение с ранней остановкой
    best_loss = float('inf')
    best_f1 = 0
    train_losses = []
    val_losses = []
    no_improve = 0
    patience = 5

    for epoch in range(EPOCHS):
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
        if val_loss < best_loss or val_f1 > best_f1:
            if val_loss < best_loss:
                best_loss = val_loss
            if val_f1 > best_f1:
                best_f1 = val_f1
            no_improve = 0
            torch.save(model.state_dict(), 'best_triplet_model.pth')
            print(f"Saved new best model (loss: {val_loss:.4f}, F1: {val_f1:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                notify(f"Early stopping at epoch {epoch + 1}")
                break

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


if __name__ == '__main__':
    try:
        notify("Эксперимент начался")
        main()
        notify("Эксперимент закончился")
    except Exception as e:
        notify(str(e))
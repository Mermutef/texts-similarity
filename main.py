import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import warnings

warnings.filterwarnings('ignore')

import logging

logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

import re
import os
import csv
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             precision_score, recall_score)
import matplotlib.pyplot as plt
from tqdm import tqdm
from extensions import notify
import gc


# Настройки воспроизводимости
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Очистка памяти GPU
def clear_gpu_memory():
    torch.cuda.empty_cache()
    gc.collect()


# Функция очистки текста с обработкой NaN
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower().strip()
    text = re.sub(r"[^a-zа-яё0-9\s.,!?;:]+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text if text else "empty"


# Триплетный датасет с обработкой дисбаланса
class TripletDataset(Dataset):
    def __init__(self, data_dict, tokenizer, max_length=128, neg_samples=3):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data_dict = data_dict
        self.question_ids = list(data_dict.keys())
        self.neg_samples = neg_samples
        self.samples = self._prepare_samples()

        # Статистика классов
        self.labels = [s['mark'] for s in self.samples]
        print(
            f"Dataset created: {len(self)} samples | Positive: {sum(self.labels)} | Negative: {len(self.labels) - sum(self.labels)}")

    def _prepare_samples(self):
        samples = []
        for q_id, df in self.data_dict.items():
            true_answer = clean_text(str(df['true_answer'].iloc[0]))
            for _, row in df.iterrows():
                samples.append({
                    'question_id': q_id,
                    'answer': clean_text(str(row['answer'])),
                    'true_answer': true_answer,
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

        # Выбираем отрицательные примеры (с учетом дисбаланса)
        negative_texts = []
        for _ in range(self.neg_samples):
            if random.random() < 0.8:  # 80% вероятность взять реальный негативный ответ
                other_qid = random.choice([q for q in self.question_ids if q != q_id])
                other_df = self.data_dict[other_qid]
                negative_sample = other_df.sample(1).iloc[0]
                negative_texts.append(clean_text(str(negative_sample['answer'])))
            else:  # 20% вероятность сгенерировать случайный текст
                words = anchor_text.split()
                if len(words) > 0:
                    random_text = " ".join(random.choices(words, k=min(10, len(words))))
                else:
                    random_text = "empty"
                negative_texts.append(clean_text(random_text))

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

        negative_encs = []
        for neg_text in negative_texts:
            enc = self.tokenizer(
                neg_text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            negative_encs.append({
                'negative_ids': enc['input_ids'].squeeze(0),
                'negative_mask': enc['attention_mask'].squeeze(0)
            })

        return {
            'anchor_ids': anchor_enc['input_ids'].squeeze(0),
            'anchor_mask': anchor_enc['attention_mask'].squeeze(0),
            'positive_ids': positive_enc['input_ids'].squeeze(0),
            'positive_mask': positive_enc['attention_mask'].squeeze(0),
            'negative_encs': negative_encs,
            'label': sample['mark']
        }


# Функция для объединения данных в батчи
def collate_fn(batch):
    anchor_ids = torch.stack([item['anchor_ids'] for item in batch])
    anchor_mask = torch.stack([item['anchor_mask'] for item in batch])
    positive_ids = torch.stack([item['positive_ids'] for item in batch])
    positive_mask = torch.stack([item['positive_mask'] for item in batch])
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.float)

    # Обработка негативов
    neg_samples = len(batch[0]['negative_encs'])
    negative_encs = []
    for i in range(neg_samples):
        neg_ids = torch.stack([item['negative_encs'][i]['negative_ids'] for item in batch])
        neg_mask = torch.stack([item['negative_encs'][i]['negative_mask'] for item in batch])
        negative_encs.append({
            'negative_ids': neg_ids,
            'negative_mask': neg_mask
        })

    return {
        'anchor_ids': anchor_ids,
        'anchor_mask': anchor_mask,
        'positive_ids': positive_ids,
        'positive_mask': positive_mask,
        'negative_encs': negative_encs,
        'label': labels
    }


# Модель с улучшенной архитектурой
class TripletBERT(nn.Module):
    def __init__(self, model_name='distilbert-base-multilingual-cased'):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(model_name)

        # Частичная заморозка (только первые 3 слоя)
        for i, layer in enumerate(self.bert.transformer.layer):
            if i < 3:
                for param in layer.parameters():
                    param.requires_grad = False

        # Проекция в пространство меньшей размерности
        self.projection = nn.Sequential(
            nn.Linear(self.bert.config.dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state[:, 0, :]
        return self.projection(embeddings)

    def normalize(self, embeddings):
        return F.normalize(embeddings, p=2, dim=1)


# Triplet Loss с динамическим margin
class TripletLoss(nn.Module):
    def __init__(self, base_margin=0.5, hard_margin_factor=1.5):
        super().__init__()
        self.base_margin = base_margin
        self.hard_margin_factor = hard_margin_factor

    def forward(self, anchor, positive, negatives):
        # Положительное расстояние
        pos_dist = F.pairwise_distance(anchor, positive, p=2)

        # Отрицательные расстояния
        neg_dists = [F.pairwise_distance(anchor, neg, p=2) for neg in negatives]
        neg_dists = torch.stack(neg_dists, dim=1)

        # Выбор самого сложного негатива
        hardest_neg_dist, _ = neg_dists.min(dim=1)

        # Динамический margin для сложных примеров
        margin = self.base_margin + (self.hard_margin_factor * (1 - hardest_neg_dist.detach()))

        losses = F.relu(pos_dist - hardest_neg_dist + margin)
        return losses.mean(), pos_dist.mean(), hardest_neg_dist.mean()


# Функции для обучения
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    pos_dists = []
    neg_dists = []

    for batch in tqdm(dataloader, desc="Training"):
        anchor_ids = batch['anchor_ids'].to(device)
        anchor_mask = batch['anchor_mask'].to(device)
        positive_ids = batch['positive_ids'].to(device)
        positive_mask = batch['positive_mask'].to(device)
        negative_encs = batch['negative_encs']

        # Перемещаем негативы на устройство
        neg_ids_list = [enc['negative_ids'].to(device) for enc in negative_encs]
        neg_mask_list = [enc['negative_mask'].to(device) for enc in negative_encs]

        # Очистка памяти перед вычислениями
        clear_gpu_memory()

        # Прямое распространение
        anchor_emb = model(anchor_ids, anchor_mask)
        positive_emb = model(positive_ids, positive_mask)
        negative_embs = [model(neg_ids, neg_mask) for neg_ids, neg_mask in zip(neg_ids_list, neg_mask_list)]

        # Нормализация
        anchor_emb = model.normalize(anchor_emb)
        positive_emb = model.normalize(positive_emb)
        negative_embs = [model.normalize(neg) for neg in negative_embs]

        # Вычисление потерь
        optimizer.zero_grad()
        loss, pos_dist, neg_dist = criterion(anchor_emb, positive_emb, negative_embs)

        # Обратное распространение
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        pos_dists.append(pos_dist.item())
        neg_dists.append(neg_dist.item())

        # Очистка промежуточных переменных
        del anchor_emb, positive_emb, negative_embs
        clear_gpu_memory()

    avg_pos_dist = sum(pos_dists) / len(pos_dists)
    avg_neg_dist = sum(neg_dists) / len(neg_dists)
    return total_loss / len(dataloader), avg_pos_dist, avg_neg_dist


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    pos_dists = []
    neg_dists = []
    all_labels = []
    all_similarities = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            anchor_ids = batch['anchor_ids'].to(device)
            anchor_mask = batch['anchor_mask'].to(device)
            positive_ids = batch['positive_ids'].to(device)
            positive_mask = batch['positive_mask'].to(device)
            negative_encs = batch['negative_encs']
            labels = batch['label'].tolist()

            # Перемещаем негативы на устройство
            neg_ids_list = [enc['negative_ids'].to(device) for enc in negative_encs]
            neg_mask_list = [enc['negative_mask'].to(device) for enc in negative_encs]

            # Прямое распространение
            anchor_emb = model(anchor_ids, anchor_mask)
            positive_emb = model(positive_ids, positive_mask)
            negative_embs = [model(neg_ids, neg_mask) for neg_ids, neg_mask in zip(neg_ids_list, neg_mask_list)]

            # Нормализация
            anchor_emb = model.normalize(anchor_emb)
            positive_emb = model.normalize(positive_emb)
            negative_embs = [model.normalize(neg) for neg in negative_embs]

            # Вычисление потерь
            loss, pos_dist, neg_dist = criterion(anchor_emb, positive_emb, negative_embs)
            total_loss += loss.item()
            pos_dists.append(pos_dist.item())
            neg_dists.append(neg_dist.item())

            # Вычисление сходства для метрик
            pos_similarity = F.cosine_similarity(anchor_emb, positive_emb)
            neg_similarities = [F.cosine_similarity(anchor_emb, neg) for neg in negative_embs]

            all_labels.extend([1] * len(pos_similarity))
            all_similarities.extend(pos_similarity.cpu().numpy())

            for neg_sim in neg_similarities:
                all_labels.extend([0] * len(neg_sim))
                all_similarities.extend(neg_sim.cpu().numpy())

            # Очистка промежуточных переменных
            del anchor_emb, positive_emb, negative_embs
            clear_gpu_memory()

    # Вычисление метрик
    if len(set(all_labels)) < 2:
        print("Warning: Only one class present in evaluation")
        return (0, 0, 0, 0, 0, 0, 0, 0, 0.5, [])

    thresholds = np.linspace(0.1, 0.9, 50)
    f1_scores = []
    for t in thresholds:
        try:
            f1 = f1_score(all_labels, (np.array(all_similarities) > t).astype(int))
            f1_scores.append(f1)
        except:
            f1_scores.append(0)

    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]

    bin_preds = (np.array(all_similarities) > best_threshold).astype(int)
    acc = accuracy_score(all_labels, bin_preds)
    precision = precision_score(all_labels, bin_preds, zero_division=0)
    recall = recall_score(all_labels, bin_preds, zero_division=0)
    auc = roc_auc_score(all_labels, all_similarities) if len(set(all_labels)) > 1 else 0.5

    avg_loss = total_loss / len(dataloader)
    avg_pos_dist = sum(pos_dists) / len(pos_dists)
    avg_neg_dist = sum(neg_dists) / len(neg_dists)

    return (avg_loss, avg_pos_dist, avg_neg_dist,
            acc, best_f1, precision, recall, auc,
            best_threshold, all_similarities)


# Функция для сохранения метрик
def save_metrics(epoch, train_metrics, val_metrics, filename="training_metrics.csv"):
    train_loss, train_pos_dist, train_neg_dist = train_metrics
    val_loss, val_pos_dist, val_neg_dist, val_acc, val_f1, val_precision, val_recall, val_auc, val_threshold, _ = val_metrics

    metrics = {
        'epoch': epoch,
        'train_loss': train_loss,
        'train_pos_dist': train_pos_dist,
        'train_neg_dist': train_neg_dist,
        'val_loss': val_loss,
        'val_pos_dist': val_pos_dist,
        'val_neg_dist': val_neg_dist,
        'val_acc': val_acc,
        'val_f1': val_f1,
        'val_precision': val_precision,
        'val_recall': val_recall,
        'val_auc': val_auc,
        'val_threshold': val_threshold
    }

    file_exists = os.path.isfile(filename)
    with open(filename, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=metrics.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics)


# Функция для построения графиков
def plot_metrics(metrics_file="training_metrics.csv"):
    if not os.path.exists(metrics_file):
        print(f"Metrics file {metrics_file} not found")
        return

    df = pd.read_csv(metrics_file)

    plt.figure(figsize=(15, 15))

    # График потерь
    plt.subplot(3, 2, 1)
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss')
    plt.plot(df['epoch'], df['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Triplet Loss')
    plt.legend()
    plt.grid(True)

    # График расстояний
    plt.subplot(3, 2, 2)
    plt.plot(df['epoch'], df['train_pos_dist'], label='Positive Distance')
    plt.plot(df['epoch'], df['train_neg_dist'], label='Negative Distance')
    plt.plot(df['epoch'], df['val_pos_dist'], '--', label='Val Positive')
    plt.plot(df['epoch'], df['val_neg_dist'], '--', label='Val Negative')
    plt.xlabel('Epoch')
    plt.ylabel('Distance')
    plt.title('Embedding Distances')
    plt.legend()
    plt.grid(True)

    # График F1 и AUC
    plt.subplot(3, 2, 3)
    plt.plot(df['epoch'], df['val_f1'], label='F1 Score')
    plt.plot(df['epoch'], df['val_auc'], label='AUC')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Validation F1 and AUC')
    plt.legend()
    plt.grid(True)

    # График точности и полноты
    plt.subplot(3, 2, 4)
    plt.plot(df['epoch'], df['val_precision'], label='Precision')
    plt.plot(df['epoch'], df['val_recall'], label='Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Validation Precision and Recall')
    plt.legend()
    plt.grid(True)

    # График точности
    plt.subplot(3, 2, 5)
    plt.plot(df['epoch'], df['val_acc'], label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.grid(True)

    # График порога
    plt.subplot(3, 2, 6)
    plt.plot(df['epoch'], df['val_threshold'], label='Threshold')
    plt.xlabel('Epoch')
    plt.ylabel('Threshold')
    plt.title('Optimal Similarity Threshold')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_metrics_plot.png')
    plt.close()


# Основной пайплайн
def main():
    # Конфигурация
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    BATCH_SIZE = 12  # уменьшено для видеопамяти
    EPOCHS = 100
    LR = 5e-6
    NEG_SAMPLES = 4  # несколько негативов на один пример

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
        file_path = os.path.join(dir_path, file_name)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            data_dict[q_id] = df
        else:
            print(f"Warning: File not found {file_path}")

    # Разделение на train/val
    all_questions = list(data_dict.keys())
    train_questions, val_questions = train_test_split(
        all_questions, test_size=0.2, random_state=42
    )

    train_dict = {q: data_dict[q] for q in train_questions}
    val_dict = {q: data_dict[q] for q in val_questions}

    print(f"Train questions: {train_questions}")
    print(f"Val questions: {val_questions}")

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')

    # Создание датасетов
    train_dataset = TripletDataset(train_dict, tokenizer, neg_samples=NEG_SAMPLES)
    val_dataset = TripletDataset(val_dict, tokenizer, neg_samples=NEG_SAMPLES)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2
    )

    # Инициализация модели
    model = TripletBERT().to(device)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    criterion = TripletLoss(base_margin=0.5, hard_margin_factor=1.5)

    # Обучение
    best_f1 = 0
    best_model_path = 'best_model.pth'

    for epoch in range(EPOCHS):
        print(f"\n{'=' * 20} Epoch {epoch + 1}/{EPOCHS} {'=' * 20}")
        clear_gpu_memory()

        try:
            # Обучение
            train_loss, train_pos_dist, train_neg_dist = train_epoch(
                model, train_loader, optimizer, criterion, device
            )

            # Валидация
            val_metrics = evaluate(model, val_loader, criterion, device)
            val_loss, val_pos_dist, val_neg_dist, val_acc, val_f1, val_precision, val_recall, val_auc, val_threshold, _ = val_metrics

            # Сохранение метрик
            save_metrics(epoch + 1,
                         (train_loss, train_pos_dist, train_neg_dist),
                         val_metrics)

            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"Train Distances: Pos {train_pos_dist:.4f} | Neg {train_neg_dist:.4f}")
            print(f"Val Distances: Pos {val_pos_dist:.4f} | Neg {val_neg_dist:.4f}")
            print(f"Val Metrics: Acc {val_acc:.4f} | F1 {val_f1:.4f} | AUC {val_auc:.4f}")
            print(f"Val Precision: {val_precision:.4f} | Recall {val_recall:.4f} | Threshold {val_threshold:.4f}")

            # Уведомление
            if epoch % 10 == 0:
                notify(f"Epoch {epoch + 1}/{EPOCHS}\n"
                       f"Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}\n"
                       f"Distances: Pos {val_pos_dist:.4f} | Neg {val_neg_dist:.4f}")

            # Сохранение лучшей модели
            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'val_f1': val_f1,
                    'threshold': val_threshold
                }, best_model_path)
                print(f"Saved best model with F1: {val_f1:.4f}")

        except Exception as e:
            print(f"Error during epoch {epoch + 1}: {str(e)}")
            notify(f"Error during epoch {epoch + 1}: {str(e)}")
            break

    # Построение графиков
    plot_metrics()

    # Загрузка лучшей модели
    if os.path.exists(best_model_path):
        try:
            # Исправление для PyTorch 2.6+
            checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            threshold = checkpoint['threshold']
            print(f"Loaded best model from epoch {checkpoint['epoch']} with F1: {checkpoint['val_f1']:.4f}")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            notify(f"Error loading model: {str(e)}")
            return

    # Финальная оценка
    try:
        val_metrics = evaluate(model, val_loader, criterion, device)
        _, _, _, val_acc, val_f1, val_precision, val_recall, val_auc, _, val_similarities = val_metrics

        print("\nFinal Evaluation:")
        print(f"Accuracy: {val_acc:.4f} | F1: {val_f1:.4f} | AUC: {val_auc:.4f}")
        print(f"Precision: {val_precision:.4f} | Recall: {val_recall:.4f}")
        notify("Training completed\n"
               f"Final F1: {val_f1:.4f} | AUC: {val_auc:.4f}\n"
               f"Precision: {val_precision:.4f} | Recall: {val_recall:.4f}")

        # Визуализация распределения сходств
        if val_similarities:
            plt.figure(figsize=(10, 6))
            plt.hist(val_similarities, bins=50, alpha=0.7)
            plt.axvline(x=threshold, color='r', linestyle='--', label=f'Threshold: {threshold:.2f}')
            plt.xlabel('Cosine Similarity')
            plt.ylabel('Frequency')
            plt.title('Distribution of Cosine Similarities')
            plt.legend()
            plt.savefig('similarity_distribution.png')
    except Exception as e:
        print(f"Error during final evaluation: {str(e)}")
        notify(f"Error during final evaluation: {str(e)}")


if __name__ == '__main__':
    try:
        notify("Эксперимент начался")
        main()
        notify("Эксперимент закончился")
    except Exception as e:
        notify(f"Ошибка: {str(e)}")
        raise e

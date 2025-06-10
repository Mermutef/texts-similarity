import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import os
import csv
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoModel, AutoTokenizer
import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from extensions import notify

# Конфигурация
MODEL_NAME = 'slone/LaBSE-en-ru-myv-v1'
BATCH_SIZE = 12
MAX_LENGTH = 128
NUM_EPOCHS = 200
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = 'corpus'
SAVE_DIR = './'
os.makedirs(SAVE_DIR, exist_ok=True)
SEED = 579566
MARGIN = 0.8  # Оптимально для текстовой схожести
DROPOUT_RATE = 0.2  # Умеренный dropout
LEARNING_RATE = 3e-5  # Компромисс между скоростью и стабильностью
SCALE = 1.0  # Без масштабирования потерь
WEIGHT_DECAY = 1e-5  # Слабая L2 регуляризация
PATIENCE = 5
ACTIVE_LAYERS = 6


# Настройки воспроизводимости
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed()


def get_file_sizes(data_dir):
    """Возвращает размеры (количество строк) для каждого файла вопросов"""
    file_sizes = []
    for i in range(8):
        with open(os.path.join(data_dir, f'question_{i}.csv'), 'r', encoding='utf-8') as f:
            file_sizes.append(sum(1 for _ in f) - 1)  # -1 для заголовка
    return file_sizes


def split_file_indices(file_sizes, min_val_files=2, target_train_ratio=0.8):
    """Разделяет индексы файлов на тренировочные и валидационные с гарантией min_val_files в валидации"""
    n_files = len(file_sizes)
    indices = list(range(n_files))
    random.shuffle(indices)

    # Начинаем с минимального количества файлов в валидации
    val_indices = indices[:min_val_files]
    train_indices = indices[min_val_files:]

    # Рассчитываем текущее распределение данных
    total_examples = sum(file_sizes)
    train_size = sum(file_sizes[i] for i in train_indices)
    current_ratio = train_size / total_examples

    # Добавляем дополнительные файлы в валидацию, если тренировочных данных слишком много
    idx = min_val_files
    while current_ratio > target_train_ratio and idx < n_files:
        val_indices.append(indices[idx])
        train_indices.remove(indices[idx])
        train_size = sum(file_sizes[i] for i in train_indices)
        current_ratio = train_size / total_examples
        idx += 1

    # Если после добавления всех файлов в валидацию соотношение все еще не достигнуто
    if current_ratio > target_train_ratio:
        # Перераспределяем часть данных внутри файлов
        for i in train_indices:
            if file_sizes[i] > 0:
                # Рассчитываем сколько примеров нужно переместить в валидацию
                needed_reduction = int((current_ratio - target_train_ratio) * total_examples)
                if needed_reduction > 0:
                    # Помечаем файл как частично валидационный
                    val_indices.append(i)
                    # Рассчитываем сколько примеров оставить в тренировочном наборе
                    keep_in_train = max(1, file_sizes[i] - needed_reduction)
                    file_sizes[i] = keep_in_train
                    break

    return train_indices, val_indices


def load_and_preprocess_data(data_dir, train_indices, val_indices):
    """Загрузка данных с разделением на тренировочные и валидационные файлы"""
    train_data = []
    val_data = []

    for i in range(8):
        if i in train_indices or i in val_indices:
            with open(os.path.join(data_dir, f'question_{i}.csv'), 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    item = {
                        'answer': row['answer'],
                        'true_answer': row['true_answer'],
                        'target': 1 - int(row['mark'])  # Инверсия меток
                    }

                    if i in train_indices:
                        train_data.append(item)
                    else:
                        val_data.append(item)

    random.shuffle(train_data)
    random.shuffle(val_data)
    return train_data, val_data


class SiameseDataset(Dataset):
    """Датасет для сиамской сети"""

    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        enc1 = self.tokenizer(
            item['true_answer'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        enc2 = self.tokenizer(
            item['answer'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids1': enc1['input_ids'].squeeze(0),
            'attention_mask1': enc1['attention_mask'].squeeze(0),
            'input_ids2': enc2['input_ids'].squeeze(0),
            'attention_mask2': enc2['attention_mask'].squeeze(0),
            'target': torch.tensor(item['target'], dtype=torch.float)
        }


class SiameseNetwork(nn.Module):
    def __init__(self, dropout_rate=0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(MODEL_NAME)

        # Заморозка всех слоев
        for param in self.bert.parameters():
            param.requires_grad = False

        # Разморозка верхних слоев
        for layer in self.bert.encoder.layer[-ACTIVE_LAYERS:]:
            for param in layer.parameters():
                param.requires_grad = True

        self.dropout = nn.Dropout(dropout_rate)

    def forward_once(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        # Нормализация эмбеддингов
        cls_embedding = F.normalize(cls_embedding, p=2, dim=1)
        return self.dropout(cls_embedding)

    def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2):
        output1 = self.forward_once(input_ids1, attention_mask1)
        output2 = self.forward_once(input_ids2, attention_mask2)
        return output1, output2


# Модифицированная функция потерь
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.8, scale=1.0):
        super().__init__()
        self.margin = margin
        self.scale = scale

    def forward(self, output1, output2, target):
        # Вычисляем косинусное расстояние как 1 - cosine_similarity
        cosine_sim = F.cosine_similarity(output1, output2)
        distance = 1 - cosine_sim
        loss = torch.mean(
            (1 - target) * torch.pow(distance, 2) +
            target * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
        )
        return self.scale * loss


def create_data_loaders(train_data, val_data, tokenizer):
    """Создание DataLoader'ов для готовых наборов данных"""
    train_dataset = SiameseDataset(train_data, tokenizer, MAX_LENGTH)
    val_dataset = SiameseDataset(val_data, tokenizer, MAX_LENGTH)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True
    )
    return train_loader, val_loader


def train_epoch(model, loader, loss_fn, optimizer, device):
    """Одна эпоха обучения"""
    model.train()
    epoch_loss = 0
    all_distances = []
    all_targets = []

    for batch in tqdm(loader, desc="Training"):
        input_ids1 = batch['input_ids1'].to(device)
        attention_mask1 = batch['attention_mask1'].to(device)
        input_ids2 = batch['input_ids2'].to(device)
        attention_mask2 = batch['attention_mask2'].to(device)
        targets = batch['target'].to(device)

        optimizer.zero_grad()
        output1, output2 = model(input_ids1, attention_mask1, input_ids2, attention_mask2)
        loss = loss_fn(output1, output2, targets)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        with torch.no_grad():
            dist = F.pairwise_distance(output1, output2, p=2)
            all_distances.extend(dist.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    return epoch_loss / len(loader), np.array(all_distances), np.array(all_targets)


def evaluate(model, loader, loss_fn, device):
    """Оценка модели"""
    model.eval()
    epoch_loss = 0
    all_distances = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids1 = batch['input_ids1'].to(device)
            attention_mask1 = batch['attention_mask1'].to(device)
            input_ids2 = batch['input_ids2'].to(device)
            attention_mask2 = batch['attention_mask2'].to(device)
            targets = batch['target'].to(device)

            output1, output2 = model(input_ids1, attention_mask1, input_ids2, attention_mask2)
            loss = loss_fn(output1, output2, targets)

            epoch_loss += loss.item()
            dist = F.pairwise_distance(output1, output2, p=2)
            all_distances.extend(dist.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    return epoch_loss / len(loader), np.array(all_distances), np.array(all_targets)


def compute_metrics(distances, targets):
    """Вычисление метрик"""
    if len(np.unique(targets)) < 2:
        return {'auc': 0.5, 'accuracy': 0.5, 'precision': 0.5, 'recall': 0.5, 'f1': 0.5, 'threshold': 0.5, }

    auc = roc_auc_score(targets, distances)

    # Поиск оптимального порога для F1
    thresholds = np.linspace(distances.min(), distances.max(), 100)
    best_f1 = 0
    best_metrics = {}

    for th in thresholds:
        preds = (distances > th).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets, preds, average='binary', zero_division=0
        )
        if f1 > best_f1:
            best_f1 = f1
            best_metrics = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'accuracy': accuracy_score(targets, preds),
                'threshold': th,
            }

    best_metrics['auc'] = auc
    return best_metrics


def save_metrics(epoch, train_metrics, val_metrics, filename):
    """Сохранение метрик в CSV"""
    metrics = {
        'epoch': epoch,
        'train_loss': train_metrics['loss'],
        'train_auc': train_metrics['metrics']['auc'],
        'train_f1': train_metrics['metrics']['f1'],
        'train_precision': train_metrics['metrics']['precision'],
        'train_recall': train_metrics['metrics']['recall'],
        'train_threshold': train_metrics['metrics']['threshold'],
        'val_loss': val_metrics['loss'],
        'val_auc': val_metrics['metrics']['auc'],
        'val_f1': val_metrics['metrics']['f1'],
        'val_precision': val_metrics['metrics']['precision'],
        'val_recall': val_metrics['metrics']['recall'],
        'val_threshold': val_metrics['metrics']['threshold'],
    }

    df = pd.DataFrame([metrics])

    # Проверяем, существует ли файл и содержит ли данные
    file_exists = os.path.exists(filename)
    if file_exists:
        try:
            existing_df = pd.read_csv(filename)
            if 'epoch' not in existing_df.columns:
                # Пересоздаем файл если он поврежден
                df.to_csv(filename, index=False)
                return
        except:
            # Файл поврежден - пересоздаем
            df.to_csv(filename, index=False)
            return

    header = not file_exists
    df.to_csv(filename, mode='a', header=header, index=False)


def plot_metrics(metrics_file, save_dir):
    """Построение графиков метрик"""
    if not os.path.exists(metrics_file):
        print(f"Metrics file {metrics_file} not found")
        return

    try:
        df = pd.read_csv(metrics_file)
        if df.empty:
            print("Metrics file is empty")
            return

        # Проверка наличия необходимых колонок
        required_columns = {'epoch', 'train_loss', 'train_f1', 'val_loss', 'val_f1',
                            'val_precision', 'val_recall', 'train_auc', 'val_auc'}
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            print(f"Missing columns in metrics file: {missing}")
            return

        plt.figure(figsize=(15, 10))

        # График потерь
        plt.subplot(2, 2, 1)
        plt.plot(df['epoch'], df['train_loss'], label='Train Loss')
        plt.plot(df['epoch'], df['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss')
        plt.legend()
        plt.grid(True)

        # График F1
        plt.subplot(2, 2, 2)
        plt.plot(df['epoch'], df['train_f1'], label='Train F1')
        plt.plot(df['epoch'], df['val_f1'], label='Validation F1')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.title('F1 Score')
        plt.legend()
        plt.grid(True)

        # График точности и полноты (только валидация)
        plt.subplot(2, 2, 3)
        plt.plot(df['epoch'], df['val_precision'], label='Validation Precision')
        plt.plot(df['epoch'], df['val_recall'], label='Validation Recall')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.title('Validation Precision and Recall')
        plt.legend()
        plt.grid(True)

        # График AUC
        plt.subplot(2, 2, 4)
        plt.plot(df['epoch'], df['train_auc'], label='Train AUC')
        plt.plot(df['epoch'], df['val_auc'], label='Validation AUC')
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.title('ROC AUC')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()

        plot_filename = os.path.join(save_dir, 'training_metrics_plot.png')
        plt.savefig(plot_filename)
        plt.close()
        print(f"Saved metrics plot to {plot_filename}")

    except Exception as e:
        print(f"Error plotting metrics: {str(e)}")
        import traceback
        traceback.print_exc()


def main():
    # Инициализация
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    file_sizes = get_file_sizes(DATA_DIR)

    train_indices, val_indices = split_file_indices(file_sizes)
    print(f"Train files: {train_indices}")
    print(f"Validation files: {val_indices}")

    train_data, val_data = load_and_preprocess_data(DATA_DIR, train_indices, val_indices)

    print(f"Train samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")

    train_loader, val_loader = create_data_loaders(train_data, val_data, tokenizer)

    # Модель и оптимизатор
    model = SiameseNetwork(dropout_rate=DROPOUT_RATE).to(DEVICE)
    loss_fn = ContrastiveLoss(margin=MARGIN, scale=SCALE)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=1e-5,
        max_lr=3e-5,
        step_size_up=500,
        mode='exp_range',
        gamma=0.99
    )

    # Ранняя остановка с несколькими критериями
    best_val_f1 = 0
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    best_model = None
    min_overfit_gap = float('inf')  # минимальный разрыв между train и val F1
    overfit_gap_threshold = 0.05  # допустимый разрыв между train и val F1

    # Файл для метрик
    metrics_file = os.path.join(SAVE_DIR, 'training_metrics.csv')

    # Удаление предыдущего файла метрик перед началом эксперимента
    if os.path.exists(metrics_file):
        try:
            os.remove(metrics_file)
            print(f"Removed existing metrics file: {metrics_file}")
        except Exception as e:
            print(f"Error removing metrics file: {str(e)}")

    # Обучение
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")

        # Тренировка
        train_loss, train_dists, train_targets = train_epoch(
            model, train_loader, loss_fn, optimizer, DEVICE
        )
        train_metrics = compute_metrics(train_dists, train_targets)

        # Валидация
        val_loss, val_dists, val_targets = evaluate(
            model, val_loader, loss_fn, DEVICE
        )
        val_metrics = compute_metrics(val_dists, val_targets)

        # Логирование
        save_metrics(
            epoch,
            {'loss': train_loss, 'metrics': train_metrics},
            {'loss': val_loss, 'metrics': val_metrics},
            metrics_file
        )

        print(f"Train Loss: {train_loss:.4f} | F1: {train_metrics['f1']:.4f} | AUC: {train_metrics['auc']:.4f}")
        print(f"Val Loss: {val_loss:.4f} | F1: {val_metrics['f1']:.4f} | AUC: {val_metrics['auc']:.4f}")

        # Уведомление в телеграм
        if epoch % 10 == 0:
            notify(
                f"Epoch {epoch}\n"
                f"Train Loss: {train_loss:.4f}, F1: {train_metrics['f1']:.4f}\n"
                f"Val Loss: {val_loss:.4f}, F1: {val_metrics['f1']:.4f}"
            )

        # Обновление после каждого батча
        scheduler.step()

        # Вычисление метрик для ранней остановки
        current_val_f1 = val_metrics['f1']
        overfit_gap = train_metrics['f1'] - current_val_f1

        # Проверка критериев улучшения
        improvement = False

        # Критерий 1: Улучшение F1 на валидации
        if current_val_f1 > best_val_f1:
            best_val_f1 = current_val_f1
            best_val_loss = val_loss
            best_epoch = epoch
            min_overfit_gap = overfit_gap
            improvement = True

        # Критерий 2: Улучшение потерь на валидации при сравнимом F1
        elif (val_loss < best_val_loss and
              abs(current_val_f1 - best_val_f1) < 0.005 and
              overfit_gap < min_overfit_gap + 0.01):
            best_val_loss = val_loss
            min_overfit_gap = overfit_gap
            improvement = True

        # Сохранение лучшей модели
        if improvement:
            patience_counter = 0
            best_model = model.state_dict()
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'best_model.pth'))
            print(f"New best model at epoch {epoch} | Val F1: {best_val_f1:.4f}")
        else:
            patience_counter += 1

        # Дополнительные критерии остановки
        stop_training = False
        stop_reasons = []

        # 1. Исчерпано терпение
        if patience_counter >= PATIENCE:
            stop_training = True
            stop_reasons.append(f"Patience exhausted ({PATIENCE} epochs)")

        # # 2. Увеличивается разрыв между train и val F1
        # if overfit_gap > min_overfit_gap + 0.05:
        #     stop_training = True
        #     stop_reasons.append(f"Overfit gap increased from {min_overfit_gap:.4f} to {overfit_gap:.4f}")
        #
        # # 3. Слишком большой абсолютный разрыв
        # if overfit_gap > overfit_gap_threshold:
        #     stop_training = True
        #     stop_reasons.append(f"Overfit gap too large ({overfit_gap:.4f} > {overfit_gap_threshold})")

        # 4. Увеличиваются потери на валидации
        if val_loss > best_val_loss * 1.5 and epoch > 10:
            stop_training = True
            stop_reasons.append(f"Val loss increased from {best_val_loss:.4f} to {val_loss:.4f}")

        if stop_training:
            print(f"\nEarly stopping at epoch {epoch}. Reasons:")
            for reason in stop_reasons:
                print(f" - {reason}")
            print(f"Best model was at epoch {best_epoch} with Val F1: {best_val_f1:.4f}")
            model.load_state_dict(best_model)
            break

    # Финализация
    plot_metrics(metrics_file, SAVE_DIR)
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'siamese_model.pth'))
    print("Training complete!")


if __name__ == "__main__":
    notify("Эксперимент начался")
    main()
    notify("Эксперимент закончился")

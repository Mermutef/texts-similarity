import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer
from final_trainer import SiameseNetwork, MAX_LENGTH
import torch.nn.functional as F

app = Flask(__name__)

# Инициализация модели и токенизатора (один раз при запуске)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = './best_model.pth'
MODEL_NAME = 'slone/LaBSE-en-ru-myv-v1'

print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = SiameseNetwork().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("Model loaded successfully!")


def calculate_similarity(answer1, answer2):
    """Вычисляет схожесть между двумя текстовыми ответами"""
    enc1 = tokenizer(
        answer1,
        max_length=MAX_LENGTH,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    enc2 = tokenizer(
        answer2,
        max_length=MAX_LENGTH,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids1 = enc1['input_ids'].to(DEVICE)
    attention_mask1 = enc1['attention_mask'].to(DEVICE)
    input_ids2 = enc2['input_ids'].to(DEVICE)
    attention_mask2 = enc2['attention_mask'].to(DEVICE)

    with torch.no_grad():
        emb1, emb2 = model(input_ids1, attention_mask1, input_ids2, attention_mask2)

    return F.cosine_similarity(emb1, emb2, dim=1).item()


@app.route('/compare', methods=['POST'])
def compare_answers():
    """Основной эндпоинт для сравнения ответов"""
    data = request.json

    # Валидация входных данных
    required_fields = ['student_answer', 'reference_answer']
    if not all(field in data for field in required_fields):
        return jsonify({"error": "Missing required fields"}), 400

    student_answer = data['student_answer']
    reference_answer = data['reference_answer']

    # Вычисление метрик
    semantic_sim = calculate_similarity(student_answer, reference_answer)

    return jsonify({
        "similarity": semantic_sim,
    })


@app.route('/health', methods=['GET'])
def health_check():
    """Проверка работоспособности сервиса"""
    return jsonify({"status": "ok", "device": str(DEVICE)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

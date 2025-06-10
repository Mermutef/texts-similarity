import requests
import os
import json
from urllib.parse import quote
from dotenv import load_dotenv
from typing import Optional, List


def notify(
        message: str,
        env_path: Optional[str] = None,
        image_dir: Optional[str] = None,
        parse_mode: Optional[str] = None
) -> None:
    """Отправляет сообщение в Telegram с возможностью прикрепления изображений.

    Функция автоматически определяет, нужно ли отправлять только текст или текст с медиагруппой.
    Поддерживает HTML/Markdown форматирование через parse_mode.

    Args:
        message (str): Текст сообщения для отправки. Может содержать разметку согласно parse_mode.
        env_path (Optional[str]): Путь к файлу .env. Если None, ищет в текущей директории.
        image_dir (Optional[str]): Путь к директории с изображениями. Если None, отправляется только текст.
        parse_mode (Optional[str]): Режим форматирования текста ('HTML' или 'MarkdownV2').

    Raises:
        ValueError: Если не найдены TELEGRAM_BOT_TOKEN или TELEGRAM_CHAT_ID в .env файле.
        FileNotFoundError: Если указанная директория с изображениями не существует.
        Exception: При ошибках API Telegram (отправка сообщения или медиафайлов).

    Examples:
        >>> notify("Простое текстовое сообщение")
        >>> notify("Сообщение с разметкой", parse_mode="HTML")
        >>> notify("Документ с картинками", image_dir="/path/to/images")
    """
    # Загрузка .env
    env_to_load = env_path if env_path else os.path.join(os.path.dirname(__file__), '..', '.env')
    load_dotenv(env_to_load)

    token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')

    if not token or not chat_id:
        raise ValueError("TELEGRAM_BOT_TOKEN или TELEGRAM_CHAT_ID не найдены в .env")

    if not image_dir:
        send_text(message, token, chat_id, parse_mode)
        return

    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Каталог с изображениями не найден: {image_dir}")

    image_paths = [
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    if not image_paths:
        send_text(message, token, chat_id)
        return

    send_media_group(message, image_paths, token, chat_id, parse_mode)


def send_text(text: str, token: str, chat_id: str, parse_mode: Optional[str]) -> None:
    """Отправляет текстовое сообщение через Telegram Bot API.

    Args:
        text (str): Текст сообщения. Поддерживает спецсимволы и разметку.
        token (str): Токен бота (получается из TELEGRAM_BOT_TOKEN).
        chat_id (str): ID чата (получается из TELEGRAM_CHAT_ID).
        parse_mode (Optional[str]): Режим парсинга ('HTML'/'MarkdownV2'). None - обычный текст.

    Raises:
        Exception: Если API Telegram вернуло код ошибки (status_code != 200).

    Notes:
        - Для parse_mode='HTML' экранируйте спецсимволы (<, >, &) в тексте.
        - Максимальная длина сообщения - 4096 символов.
    """
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        'chat_id': chat_id,
        'text': text,
    }
    if parse_mode is not None:
        payload['parse_mode'] = parse_mode
    response = requests.post(url, data=payload)
    if response.status_code != 200:
        raise Exception(f"Ошибка отправки текста: {response.text}")


def send_media_group(text: str, image_paths: List[str], token: str, chat_id: str, parse_mode: Optional[str]) -> None:
    """Отправляет медиагруппу с предваряющим текстовым сообщением.

    Args:
        text (str): Текст сообщения, который будет отправлен отдельно перед медиагруппой.
        image_paths (List[str]): Список абсолютных путей к изображениям.
        token (str): Токен бота из TELEGRAM_BOT_TOKEN.
        chat_id (str): ID чата из TELEGRAM_CHAT_ID.
        parse_mode (Optional[str]): Режим форматирования текста.

    Raises:
        Exception: Если возникла ошибка при отправке медиагруппы.

    Notes:
        - Telegram позволяет отправлять до 10 изображений в одной медиагруппе.
        - Каждое изображение должно быть не больше 10MB.
        - Подписи к изображениям не поддерживаются в данной реализации.
    """
    # Сначала отправляем текст
    send_text(text, token, chat_id, parse_mode)

    # Затем отправляем медиагруппу без подписи
    url = f"https://api.telegram.org/bot{token}/sendMediaGroup"
    media = []
    files = {}

    for idx, image_path in enumerate(image_paths):
        media.append({
            'type': 'photo',
            'media': f"attach://photo_{idx}",
            'caption': 'null'  # Явно указываем null
        })
        files[f'photo_{idx}'] = open(image_path, 'rb')

    payload = {
        'chat_id': chat_id,
        'media': json.dumps(media, ensure_ascii=False)
    }

    try:
        response = requests.post(url, data=payload, files=files)
        if response.status_code != 200:
            raise Exception(f"Ошибка отправки медиагруппы: {response.text}")
    finally:
        for file in files.values():
            file.close()

import os
import re
from typing import Union


def ensure_directory_exists(dir_path: Union[str, os.PathLike]) -> bool:
    """Проверяет существование директории и создает ее при необходимости.

    Рекурсивно создает все недостающие директории в пути, если они не существуют.
    Если директория уже существует, функция просто возвращает `True`.

    Args:
        dir_path (Union[str, os.PathLike]): Путь к директории. Может быть строкой
            или объектом `os.PathLike` (например, `pathlib.Path`).

    Returns:
        bool: `True` - если директория существовала или была создана,
              `False` - если произошла ошибка.

    Raises:
        OSError: Если возникли проблемы с правами доступа или некорректный путь.
                 (Ловится внутри функции, но можно обработать снаружи.)

    Examples:
        >>> ensure_directory_exists("/path/to/directory")
        True
        >>> ensure_directory_exists("relative/path")
        True
    """
    try:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        return True

    except OSError as error:
        print(f"[ERROR] Не удалось создать директорию {dir_path}: {error}")
        return False


def get_next_experiment_number(directory="."):
    """Определяет следующий номер эксперимента на основе существующих папок.
    
    Функция сканирует переданную директорию и ищет подкаталоги, имена которых состоят
    только из цифр (номера экспериментов). Возвращает следующий доступный номер.
    Например, если есть папки '1', '2', '5', вернёт 6.

    Args:
        directory (str, optional): Путь к директории для сканирования. 
                                  По умолчанию '.' (текущая директория).

    Returns:
        int: Следующий номер эксперимента (max_num + 1). Если нет подходящих папок, вернёт 1.

    Raises:
        OSError: Если указанная директория недоступна для чтения.

    Examples:
        >>> get_next_experiment_number()
        1  # Если нет папок с номерами

        >>> get_next_experiment_number("experiments/")
        42  # Если последняя папка — '41'

        Пример структуры директории:
        experiments/
        ├── 1/      # Учитывается
        ├── 2/      # Учитывается
        └── data/   # Игнорируется (не число)
    """
    if not os.path.exists(directory):
        return 1
    if not os.access(directory, os.R_OK):
        raise OSError(f"Нет доступа к директории: {directory}")

    subdirs = [d for d in os.listdir(directory)
               if os.path.isdir(os.path.join(directory, d))]
    max_num = 0
    pattern = re.compile(r'^(\d+)$')  # Только цифры в имени папки

    for d in subdirs:
        if pattern.match(d):
            current_num = int(d)
            max_num = max(max_num, current_num)

    return max_num + 1


def root_dir():
    """Возвращает абсолютный путь к родительскому каталогу текущего файла"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

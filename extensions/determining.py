import random
import torch


def set_deterministic_mode(seed: int = 42) -> None:
    """Устанавливает детерминированный режим для генераторов случайных чисел.

    Настраивает PyTorch (`torch`, `cuda`, `cudnn`) и встроенный модуль `random`
    так, чтобы результаты работы были воспроизводимы при одинаковом `seed`.

    Args:
        seed (int, optional): Число, используемое для инициализации генераторов.
            По умолчанию 42.

    Returns:
        None

    Note:
        Установка `torch.backends.cudnn.deterministic = True` может замедлить
        работу нейросетей на GPU, но обеспечивает полную воспроизводимость.
    """
    # Установка seed для CPU и GPU (если CUDA доступна)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Включение детерминированных алгоритмов в cuDNN (может снизить производительность)
    torch.backends.cudnn.deterministic = True

    # Установка seed для модуля random
    random.seed(seed)
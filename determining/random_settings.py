import torch


def determine_random(seed: int) -> None:
    # установка детерминированного seed для случайных величин в torch
    torch.manual_seed(seed)
    # установка детерминированного seed для случайных величин в cud
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True  # детерминированность поведения cudnn

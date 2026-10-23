"""Small helper functions used across the project."""
import os
import random

import numpy as np
import torch

import config


def set_seed(seed: int) -> None:
    """Fix all random number generators so every run gives the same results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_dirs() -> None:
    """Create output folders if they don't exist yet."""
    for folder in (config.RESULTS, config.FIGURES):
        folder.mkdir(parents=True, exist_ok=True)
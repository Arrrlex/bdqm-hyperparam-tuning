from pathlib import Path
from typing import Dict, Sequence, List

import numpy as np
from amptorch.descriptor.GMP import GMP
from amptorch.preprocessing import AtomsToData
from tqdm.contrib import tenumerate
import re
import json
import torch

rng = np.random.default_rng()

# Path to root of bdqm-hyperparam-tuning repo
bdqm_hpopt_path = Path(__file__).resolve().parents[1]
gpus = min(1, torch.cuda.device_count())


def split_data(data: Sequence, valid_pct: float):
    """
    Split data into train & validation sets.

    `valid_pct` sets how big the validation set is; setting `valid_pct=0.1` will
    mean that 10% of the data goes into the valid dataset.

    Args:
        data: the data to split
        valid_pct: a float between 0 and 1, size of the validation split
    Returns:
        train_data: the training split
        valid_data: the validation split
    """
    n = len(data)
    indices = np.arange(n)
    n_valid = int(round(valid_pct * n))
    valid_indices = rng.choice(indices, size=n_valid, replace=False)
    train_indices = np.setdiff1d(indices, valid_indices)

    train_data = [data[i] for i in train_indices]
    valid_data = [data[i] for i in valid_indices]
    return train_data, valid_data

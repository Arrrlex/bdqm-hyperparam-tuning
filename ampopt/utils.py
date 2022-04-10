import os
import socket
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import numpy as np
import torch

rng = np.random.default_rng()

# Path to root of bdqm-hyperparam-tuning repo
ampopt_path = Path(__file__).resolve().parents[1]
gpus = min(1, torch.cuda.device_count())


def split_data(data: Sequence, valid_pct: float) -> Tuple[Sequence, Sequence]:
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


def is_login_node() -> bool:
    """Return true if current node is login node"""
    return socket.gethostname() == "login-pace-ice-1.pace.gatech.edu"


def _cast(s):
    if s.isnumeric():
        return int(s)
    try:
        return float(s)
    except ValueError:
        return s

def remove_prefix(s, prefix):
    if s.startswith(prefix):
        return s[len(prefix):]
    return s

def read_params_from_env() -> Dict[str, Any]:
    params = {}
    for k, v in os.environ.items():
        if k.startswith("param_"):
            k = k[len("param_") :]
        else:
            continue
        params[k] = _cast(v)
    return params


def parse_params(param_string, prefix) -> Dict[str, Any]:
    if not param_string:
        return {}
    params = {}
    for param_pair in param_string.split(","):
        k, v = param_pair.split("=")
        params[prefix + k] = _cast(v)
    return params

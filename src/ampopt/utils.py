import os
import socket
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

# Path to root of bdqm-hyperparam-tuning repo
ampopt_path = Path(__file__).resolve().parents[2]
gpus = min(1, torch.cuda.device_count())


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

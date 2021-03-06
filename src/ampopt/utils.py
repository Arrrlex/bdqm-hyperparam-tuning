import os
import socket
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import torch
import ase.io

# Path to root of bdqm-hyperparam-tuning repo
ampopt_path = Path(__file__).resolve().parents[2]

def read_data(fname):
    if fname.endswith(".traj"):
        return ase.io.Trajectory(fname)
    else:
        return ase.io.read(fname, ":")

@lru_cache
def num_gpus():
    return torch.cuda.device_count()


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


def parse_params(param_string, prefix="") -> Dict[str, Any]:
    if not param_string:
        return {}
    params = {}
    for param_pair in param_string.split(","):
        k, v = param_pair.split("=")
        params[prefix + k] = _cast(v)
    return params


def format_params(**params):
    return ",".join(f"{k}={v}" for k, v in sorted(params.items()))


def absolute(relpath, root="cwd"):
    """
    Return absolute path as a string.

    If `root="cwd"`, return the path relative to the current working directory.

    If `root="proj"`, return the path relative to the project root
    (`bdqm-hyperparam-tuning`).
    """

    if root == "cwd":
        root_path = Path.cwd()
    elif root == "proj":
        root_path = ampopt_path
    else:
        raise Exception(f"root={root} not allowed; must be cwd or proj")

    return str((root_path / relpath).resolve())

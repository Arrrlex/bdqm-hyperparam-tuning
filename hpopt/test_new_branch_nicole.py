"""
Script to be used with the amptorch branch https://github.com/medford-group/amptorch/tree/BDQM_VIP_2022Feb
"""
from pathlib import Path

import ase.io
import numpy as np
import torch
from amptorch.ase_utils import AMPtorch
from amptorch.trainer import AtomsTrainer
from ase import Atoms
from ase.build import molecule
from ase.calculators.emt import EMT

hpopt_root = Path(__file__).resolve().parents[1]
amptorch_root = hpopt_root.parent / "amptorch"

num_gpus = 1 if torch.cuda.is_available() else 0
print(f"Running with {num_gpus} gpus")

# read all images from the trajectory
training = ase.io.read(hpopt_root / "data/water_2d.traj", index=":")

# define sigmas
nsigmas = 10
sigmas = np.linspace(0, 2.0, nsigmas + 1, endpoint=True)[1:]
print(sigmas)

# define MCSH orders
MCSHs_index = 2
MCSHs_dict = {
    0: {
        "orders": [0],
        "sigmas": sigmas,
    },
    1: {
        "orders": [0, 1],
        "sigmas": sigmas,
    },
    2: {
        "orders": [0, 1, 2],
        "sigmas": sigmas,
    },
    3: {
        "orders": [0, 1, 2, 3],
        "sigmas": sigmas,
    },
    4: {
        "orders": [0, 1, 2, 3, 4],
        "sigmas": sigmas,
    },
    5: {
        "orders": [0, 1, 2, 3, 4, 5],
        "sigmas": sigmas,
    },
    6: {
        "orders": [0, 1, 2, 3, 4, 5, 6],
        "sigmas": sigmas,
    },
    7: {
        "orders": [0, 1, 2, 3, 4, 5, 6, 7],
        "sigmas": sigmas,
    },
    8: {
        "orders": [0, 1, 2, 3, 4, 5, 6, 7, 8],
        "sigmas": sigmas,
    },
    9: {
        "orders": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "sigmas": sigmas,
    },
}
MCSHs = MCSHs_dict[MCSHs_index]  # MCSHs is now just the order of MCSHs.


GMP = {
    "MCSHs": MCSHs,
    "atom_gaussians": {
        "H": str(amptorch_root / "examples/GMP/valence_gaussians/H_pseudodensity_2.g"),
        "O": str(amptorch_root / "examples/GMP/valence_gaussians/O_pseudodensity_4.g"),
    },
    "cutoff": 12.0,
    "solid_harmonics": True,
}

elements = ["H", "O"]

config = {
    "model": {
        "name": "singlenn",
        "get_forces": True,
        "num_layers": 3,
        "num_nodes": 10,
        "batchnorm": False,
        "activation": torch.nn.Tanh,
    },
    "optim": {
        "gpus": num_gpus,
        "force_coefficient": 0.01,
        "lr": 1e-3,
        "batch_size": 16,
        "epochs": 500,
        "loss": "mse",
        "metric": "mae",
    },
    "dataset": {
        "raw_data": training,
        "fp_scheme": "gmpordernorm",
        "fp_params": GMP,
        "elements": elements,
        "save_fps": True,
        "scaling": {"type": "normalize", "range": (0, 1)},
        "val_split": 0.1,
    },
    "cmd": {
        "debug": False,
        "run_dir": "./",
        "seed": 1,
        "identifier": "test",
        "verbose": True,
        # Weights and Biases used for logging - an account(free) is required
        "logger": False,
    },
}

torch.set_num_threads(1)
trainer = AtomsTrainer(config)
trainer.train()

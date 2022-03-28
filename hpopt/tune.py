import sys
import warnings

import hp_study
import numpy as np
import torch
from amptorch.dataset_lmdb import get_lmdb_dataset
from amptorch.trainer import AtomsTrainer
from ase.io import Trajectory
from optuna.integration.skorch import SkorchPruningCallback
from torch import nn
from utils import bdqm_hpopt_path

from sklearn.metrics import mean_absolute_error

gpus = min(1, torch.cuda.device_count())

data_path = bdqm_hpopt_path / "data"

valid_imgs = Trajectory(data_path / "valid.traj")
y_valid = np.array([img.get_potential_energy() for img in valid_imgs])

warnings.simplefilter("ignore")


def objective(trial):
    config = {
        "model": {
            "num_layers": trial.suggest_int("num_layers", 3, 30),
            "num_nodes": trial.suggest_int("num_nodes", 4, 200),
            "name": "singlenn",
            "get_forces": False,
            "batchnorm": trial.suggest_int("batchnorm", 0, 1),
            "dropout": 1,
            "dropout_rate": trial.suggest_float("dropout_rate", 0.0, 1.0),
            "initialization": "xavier",
            "activation": nn.Tanh,
        },
        "optim": {
            "gpus": gpus,
            "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
            "scheduler": {
                "policy": "StepLR",
                "params": {
                    "step_size": trial.suggest_int("lr_step_size", 1, 30, 5),
                    "gamma": trial.suggest_float("lr_gamma", 1e-5, 1e-1, log=True),
                },
            },
            "batch_size": trial.suggest_int("batch_size", 100, 500, 50),
            "loss": "mae",
            "epochs": 100,
        },
        "dataset": {
            "lmdb_path": [str(data_path / "train.lmdb")],
            "cache": "full",
            "val_split": 0.1,
        },
        "cmd": {
            "debug": False,
            "seed": 12,
            "identifier": "test",
            "dtype": "torch.DoubleTensor",
            "verbose": False,
            "custom_callback": SkorchPruningCallback(trial, "val_energy_mae"),
        },
    }

    trainer = AtomsTrainer(config)
    trainer.train()

    return mean_absolute_error(trainer.predict(valid_imgs)["energy"], y_valid)


def run_hyperparameter_optimization(n_trials, with_db):
    study = hp_study.get_or_create(with_db=with_db)
    study.optimize(objective, n_trials=n_trials)


def parse_args(n_trials):
    n_trials = int(n_trials)
    return n_trials


if __name__ == "__main__":
    n_trials = parse_args(*sys.argv[1:])
    print(f"Running hyperparam tuning with {n_trials} trials")
    print(f"Using {gpus} gpus")
    run_hyperparameter_optimization(
        n_trials=n_trials,
        with_db=True,
    )

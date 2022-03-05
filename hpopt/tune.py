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

gpus = min(1, torch.cuda.device_count())

data_path = bdqm_hpopt_path / "data"

valid_imgs = Trajectory(data_path / "valid.traj")
y_valid = np.array([img.get_potential_energy() for img in valid_imgs])
valid_feats = get_lmdb_dataset([str(data_path / "valid.lmdb")], cache_type="full")

warnings.simplefilter("ignore")


def objective(trial):
    num_layers = trial.suggest_int("num_layers", 3, 30)
    num_nodes = trial.suggest_int("num_nodes", 4, 50)

    # model params
    batchnorm = trial.suggest_int("batchnorm", 0, 1)  # False
    dropout = trial.suggest_int("dropout", 0, 1)  # False
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 1.0)  # 0.5
    initialization = "xavier"
    activation_name = trial.suggest_categorical("activation", ["tanh", "relu"])
    activation = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation_name]  # nn.Tanh

    # optim params
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)  # 1e-3
    batch_size = trial.suggest_int("batch_size", 100, 500, 50)  # 253
    loss = "mae"

    num_epochs = 1000

    config = {
        "model": {
            "num_layers": num_layers,
            "num_nodes": num_nodes,
            "name": "singlenn",
            "get_forces": False,
            "batchnorm": batchnorm,
            "dropout": dropout,
            "dropout_rate": dropout_rate,
            "initialization": initialization,
            "activation": activation,
        },
        "optim": {
            "gpus": gpus,
            "lr": lr,
            "batch_size": batch_size,
            "loss": loss,
            "epochs": num_epochs,
        },
        "dataset": {
            "lmdb_path": [str(data_path / "train.lmdb")],
            "cache": "full",
        },
        "cmd": {
            "debug": False,
            "seed": 12,
            "identifier": "test",
            "dtype": "torch.DoubleTensor",
            "verbose": False,
            "custom_callback": SkorchPruningCallback(trial, "valid_acc"),
        },
    }

    trainer = AtomsTrainer(config)
    trainer.train()

    y_pred = np.array(trainer.predict_from_feats(valid_feats)["energy"])

    return np.mean(np.abs(y_pred - y_valid))


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

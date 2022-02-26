import sys
import torch
from ase.io import Trajectory
from amptorch.trainer import AtomsTrainer
import os
import sys

import numpy as np
import optuna
import torch
from amptorch.dataset_lmdb import get_lmdb_dataset
from amptorch.trainer import AtomsTrainer
from ase.io import Trajectory
from torch import nn
from utils import bdqm_hpopt_path

import os

gpus = min(1, torch.cuda.device_count())

data_path = bdqm_hpopt_path / "data"

valid_imgs = Trajectory(data_path / "valid.traj")
y_valid = np.array([img.get_potential_energy() for img in valid_imgs])
valid_feats = get_lmdb_dataset([str(data_path / "valid.lmdb")], cache_type="full")


# To investigate:
#  - [x] Get this simple pipeline working with optuna
#  - [x] Use valid set instead of test set for hyperparam tuning
#  - [x] Use features already pre-prepared for prediction, rather than re-creating features
#    each time
#  - [x] Compare full cache vs no cache, do we notice a difference?
#  - [x] Try running on GPU
#  - [x] Try parallelizing
#  - [ ] Try parallelizing with GNU Parallel (https://docs.pace.gatech.edu/software/multiparallel/)
#  - [ ] Try dockerizing?
#  - [ ] Incorporate pruning using skorch integration
#  - [ ] Try fixing torch.DoubleTensor


def objective(trial):
    num_layers = trial.suggest_int("num_layers", 3, 8)
    num_nodes = trial.suggest_int("num_nodes", 4, 15)

    # model params
    batchnorm = False
    dropout = False
    dropout_rate = 0.5
    initialization = "xavier"
    activation = nn.Tanh

    # optim params
    lr = 1e-3
    batch_size = 253
    loss = "mae"

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
            "epochs": 100,
        },
        "dataset": {
            "lmdb_path": [str(data_path / "train.lmdb")],
            "cache": "full",
        },
        "cmd": {
            "debug": False,
            "seed": 12,
            "identifier": "test",
            "dtype": "torch.FloatTensor",
            "verbose": False,
        },
    }

    trainer = AtomsTrainer(config)
    trainer.train()

    y_pred = np.array(trainer.predict_from_feats(valid_feats)["energy"])

    return np.mean(np.abs(y_pred - y_valid))

def run_hyperparameter_optimization(n_trials, connection_string=None):
    if connection_string is None:
        study = optuna.create_study()
    else:
        study = optuna.create_study(
          study_name="distributed-amptorch-tuning", 
          storage=connection_string,
          load_if_exists=True,
        )

    study.optimize(objective, n_trials=n_trials)

    print(study.best_params)


def construct_connection_string():
    username = os.getenv("MYSQL_USERNAME")
    password = os.getenv("MYSQL_PASSWORD")
    node = os.getenv("MYSQL_NODE")
    db = os.getenv("DB")
    return f"mysql+pymysql://{username}:{password}@{node}/{db}"


def parse_args(n_trials):
    n_trials = int(n_trials)
    return n_trials


if __name__ == "__main__":
    connection_string = construct_connection_string()
    n_trials = parse_args(*sys.argv[1:])
    print(f"Using {gpus} gpus")
    run_hyperparameter_optimization(
        n_trials=n_trials,
        connection_string=connection_string,
    )

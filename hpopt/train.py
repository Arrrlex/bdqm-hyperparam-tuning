import warnings
from uuid import uuid4

import numpy as np
from amptorch.trainer import AtomsTrainer
from ase.io import Trajectory
from optuna.integration.skorch import SkorchPruningCallback
from sklearn.metrics import mean_absolute_error
from torch import nn

from hpopt.utils import bdqm_hpopt_path, gpus

data_path = bdqm_hpopt_path / "data"

valid_imgs = Trajectory(data_path / "valid.traj")
y_valid = np.array([img.get_potential_energy() for img in valid_imgs])

warnings.simplefilter("ignore")


def mk_objective(epochs, verbose):
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
                "epochs": epochs,
            },
            "dataset": {
                "lmdb_path": [str(data_path / "train.lmdb")],
                "cache": "full",
                # "val_split": 0.1,
            },
            "cmd": {
                # "debug": True, # prevents logging to checkpoints
                "seed": 12,
                "identifier": str(uuid4()),
                "dtype": "torch.DoubleTensor",
                "verbose": verbose,
                "custom_callback": SkorchPruningCallback(trial, "train_energy_mae"),
            },
        }

        trainer = AtomsTrainer(config)
        trainer.train()

        return mean_absolute_error(trainer.predict(valid_imgs)["energy"], y_valid)

    return objective

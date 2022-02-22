import torch
from ase.io import Trajectory
from amptorch.trainer import AtomsTrainer
import numpy as np
import optuna
from torch import nn

from utils import bdqm_hpopt_path

gpus = int(torch.cuda.is_available())

data_path = bdqm_hpopt_path / 'data'

test_imgs = Trajectory(data_path / "oc20_300_test.traj")
y_test = np.array([img.get_potential_energy() for img in test_imgs])


# To investigate:
#  - Get this simple pipeline working with optuna
#  - Use valid set instead of test set for hyperparam tuning
#  - Use features already pre-prepared for prediction, rather than re-creating features
#    each time
#  - Compare full cache vs no cache, do we notice a difference?
#  - Try running on GPU
#  - Try parallelizing
#  - Try dockerizing?
#  - Incorporate pruning

def objective(trial):
    num_layers = trial.suggest_int('num_layers', 3, 8)
    num_nodes = trial.suggest_int('num_nodes', 4, 15)

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

    y_pred = np.array(trainer.predict(test_imgs)["energy"])

    return np.mean(np.abs(y_pred - y_test))

def run_hyperparameter_optimization():
    study = optuna.create_study()
    study.optimize(objective, n_trials=5)

    print(study.best_params)

if __name__ == '__main__':
    run_hyperparameter_optimization()
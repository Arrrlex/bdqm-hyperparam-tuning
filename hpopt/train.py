import torch
from ase.io import Trajectory
from amptorch.trainer import AtomsTrainer
import amptorch
import numpy as np

from pathlib import Path

from utils import bdqm_hpopt_path

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

def calc_loss(**params):
    model_params = ["num_layers", "num_nodes", "batchnorm"]
    optim_params = ["lr", "batch_size", "loss", "metric"]

    config = {
        "model": {"name": "singlenn", "get_forces": False},
        "optim": {"epochs": 5, "force_coefficient": 0.},
        "dataset": {"lmdb_path": [str(data_path / "train.lmdb")], "cache": "full"},
        "cmd": {"debug": False, "seed": 12, "identifier": "test", "dtype": "torch.FloatTensor"},
    }

    for k in model_params:
        config["model"][k] = params[k]
    for k in optim_params:
        config["optim"][k] = params[k]

    trainer = AtomsTrainer(config)
    trainer.train()

    y_pred = np.array(trainer.predict(test_imgs)["energy"])

    return np.mean(np.abs(y_pred - y_test))

print(calc_loss(
    num_layers=3,
    num_nodes=10,
    batchnorm=True,
    lr=1e-3,
    batch_size=16,
    loss="mse",
    metric="mae",
))
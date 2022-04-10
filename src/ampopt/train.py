import warnings
from functools import partial
from uuid import uuid4
from shutil import rmtree
from pathlib import Path

from amptorch.trainer import AtomsTrainer
from optuna.integration.skorch import SkorchPruningCallback
from torch import nn

from ampopt.utils import num_gpus

warnings.simplefilter("ignore")

gpus = min(1, num_gpus())

def get_param_dict(params, trial, name, low, *args, **kwargs):
    """
    Get value of parameter as dictionary, either from params dictionary or from trial.

    Args:
        params: dict str -> int|float mapping param name to value
        trial: optuna trial
        name: name of param
        *args, **kwargs: arguments for trial.suggest_int or trial.suggest_float

    Returns:
        dictionary str -> int|float mapping `name` to its value.
    """
    try:
        val = params[name]
    except KeyError:
        param_type = type(low).__name__
        method = getattr(trial, f"suggest_{param_type}")
        val = method(name, low, *args, **kwargs)

    return {name: val}


def mk_objective(verbose, epochs, train_fname, **params):
    def objective(trial):
        get = partial(get_param_dict, params, trial)
        identifier = str(uuid4())
        config = {
            "model": {
                **get("num_layers", 3, 8),
                **get("num_nodes", 4, 15),
                "name": "singlenn",
                "get_forces": False,
                **get("batchnorm", 0, 1),
                "dropout": 1,
                **get("dropout_rate", 0.0, 1.0),
                "initialization": "xavier",
                "activation": nn.Tanh,
            },
            "optim": {
                "gpus": gpus,
                **get("lr", 1e-5, 1e-2, log=True),
                "scheduler": {
                    "policy": "StepLR",
                    "params": {
                        **get("step_size", 1, 30, 5),
                        **get("gamma", 1e-5, 1e-1, log=True),
                    },
                },
                **get("batch_size", 100, 500, 50),
                "loss": "mae",
                "epochs": epochs,
            },
            "dataset": {
                "lmdb_path": [train_fname],
                "cache": "full",
                "val_split": 0.1,
            },
            "cmd": {
                # "debug": True, # prevents logging to checkpoints
                "seed": 12,
                "identifier": identifier,
                "dtype": "torch.DoubleTensor",
                "verbose": verbose,
                "custom_callback": SkorchPruningCallback(trial, "train_energy_mae"),
            },
        }

        trainer = AtomsTrainer(config)
        trainer.train()

        score = trainer.net.history[-1, "val_energy_mae"]

        clean_up_checkpoints(identifier)

        return score

    return objective

def clean_up_checkpoints(identifier):
    for path in Path("checkpoints").glob(f"*{identifier}*"):
        rmtree(path)
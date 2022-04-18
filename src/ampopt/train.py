import warnings
from functools import partial
from pathlib import Path
from shutil import rmtree
from uuid import uuid4

from amptorch.trainer import AtomsTrainer
from optuna.integration.skorch import SkorchPruningCallback
from optuna.trial import FixedTrial
from torch import nn
import ase.io
from sklearn.metrics import mean_absolute_error

from ampopt.utils import absolute, num_gpus

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


def mk_objective(verbose, epochs, train_fname, valid_fname, **params):
    train_path = absolute(train_fname, root="cwd")

    def objective(trial):
        get = partial(get_param_dict, params, trial)
        identifier = str(uuid4())
        config = {
            "model": {
                **get("num_layers", 3, 8),
                **get("num_nodes", 4, 15),
                "name": "singlenn",
                "get_forces": False,
                "dropout": 1,
                **get("dropout_rate", 0.0, 1.0),
                "initialization": "xavier",
                "activation": nn.Tanh,
            },
            "optim": {
                "gpus": gpus,
                **get("lr", 1e-5, 1e-1, log=True),
                "scheduler": {
                    "policy": "StepLR",
                    "params": {
                        "step_size": 100,
                        **get("gamma", 0.1, 1.0),
                    },
                },
                "batch_size": 256,
                "loss": "mae",
                "epochs": epochs,
            },
            "dataset": {
                "lmdb_path": [train_path],
                "cache": "full",
                # "val_split": 0.1,
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

        test_data = ase.io.read(valid_fname)
        y_pred = [a["energy"] for a in trainer.predict(test_data)]
        y_true = [a.get_potential_energy() for a in test_data]

        score = mean_absolute_error(y_true, y_pred)

        # score = trainer.net.history[-1, "val_energy_mae"]

        clean_up_checkpoints(identifier)

        return score

    return objective


def eval_score(epochs, train_fname, **params):
    objective = mk_objective(verbose=True, epochs=epochs, train_fname=train_fname)
    return objective(FixedTrial(params))


def clean_up_checkpoints(identifier):
    for path in Path("checkpoints").glob(f"*{identifier}*"):
        rmtree(path)

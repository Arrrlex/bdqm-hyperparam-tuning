import os
import subprocess
from typing import Any, Dict

from ampopt.study import get_or_create_study, get_study
from ampopt.train import mk_objective
from ampopt.utils import (absolute, format_params, is_login_node, num_gpus,
                          parse_params, read_params_from_env)


def tune(
    trials: int,
    study: str,
    data: str,
    jobs: int = 1,
    pruner: str = "Hyperband",
    sampler: str = "CmaEs",
    epochs: int = 100,
    params: str = "",
    verbose: bool = False,
):
    if jobs < 1:
        print("Must be at least 1 job")
        print("Aborting")
        return

    if is_login_node():
        print("Don't run tuning on the login node!")
        print("Aborting")
        return

    if 0 < num_gpus() < jobs:
        print(
            f"Warning: running {jobs} jobs with only {num_gpus()} GPUs, trouble ahead"
        )

    print(f"Running hyperparam tuning with:")
    print(f" - study_name: {study}")
    print(f" - dataset: {data}")
    print(f" - n_trials: {trials}")
    print(f" - sampler: {sampler}")
    print(f" - pruner: {pruner}")
    print(f" - num epochs: {epochs}")

    data = absolute(data, root="cwd")

    _ = get_or_create_study(study_name=study, pruner=pruner, sampler=sampler)

    if params == "env":
        print("Reading params from env")
        params_dict = read_params_from_env()
    else:
        params_dict = parse_params(params)

    if params_dict:
        print(f" - params:")
        for k, v in params_dict.items():
            print(f"   - {k}: {v}")

    if jobs == 1:
        tune_local(
            study_name=study,
            n_epochs=epochs,
            data=data,
            n_trials=trials,
            params_dict=params_dict,
            verbose=verbose,
        )
    else:
        cmd = ["ampopt", "tune-local"]
        cmd += ["--study", study]
        cmd += ["--data", data]
        cmd += ["--trials", str(trials)]
        cmd += ["--epochs", str(epochs)]
        if params_dict:
            cmd += ["--params", format_params(**params_dict)]
        if verbose:
            cmd.append("--verbose")
        for i in range(jobs):
            subprocess.Popen(cmd, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(i)})


def tune_local(
    study_name: str,
    n_epochs: int,
    data: str,
    n_trials: int,
    params_dict: Dict[str, Any],
    verbose: bool,
):
    study = get_study(study_name=study_name)
    objective = mk_objective(
        verbose=verbose, epochs=n_epochs, train_fname=data, **params_dict
    )
    study.optimize(objective, n_trials=n_trials)

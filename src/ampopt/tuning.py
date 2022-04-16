from typing import Any, Dict

import subprocess

from ampopt.study import get_or_create_study, get_study
from ampopt.train import mk_objective
from ampopt.utils import (is_login_node, num_gpus, parse_params,
                          read_params_from_env, format_params)


def tune(
    n_jobs: int = 1,
    n_trials_per_job: int = 10,
    study_name: str = None,
    data: str = "data/oc20_3k_train.lmdb",
    pruner: str = "Median",
    sampler: str = "CmaEs",
    n_epochs: int = 100,
    params: str = "",
    verbose: bool = False,
):
    if n_jobs < 1:
        print("Must be at least 1 job")
        print("Aborting")
        return

    if is_login_node():
        print("Don't run tuning on the login node!")
        print("Aborting")
        return

    if study_name is None:
        print("study_name must be provided")
        print("Aborting")
        return

    if 0 < num_gpus() < n_jobs:
        print(
            f"Warning: running {n_jobs} jobs with only {num_gpus()} GPUs, trouble ahead"
        )

    print(f"Running hyperparam tuning with:")
    print(f" - study_name: {study_name}")
    print(f" - dataset: {data}")
    print(f" - n_trials: {n_trials_per_job}")
    print(f" - sampler: {sampler}")
    print(f" - pruner: {pruner}")
    print(f" - num epochs: {n_epochs}")

    _ = get_or_create_study(
        study_name=study_name, pruner=pruner, sampler=sampler
    )

    if params == "env":
        print("Reading params from env")
        params_dict = read_params_from_env()
    else:
        params_dict = parse_params(params)

    if params_dict:
        print(f" - params:")
        for k, v in params_dict.items():
            print(f"   - {k}: {v}")

    if n_jobs == 1:
        tune_local(
            study_name=study_name,
            n_epochs=n_epochs,
            data=data,
            n_trials=n_trials_per_job,
            params_dict=params_dict,
            verbose=verbose,
        )
    else:
        cmd = ["conda", "run", "-n", "bdqm-hpopt"]
        cmd += ["ampopt", "tune-local"]
        cmd += ["--study-name", study_name]
        cmd += ["--data", data]
        cmd += ["--n-trials-per-job", str(n_trials_per_job)]
        cmd += ["--n-epochs", str(n_epochs)]
        cmd += ["--params", format_params(**params_dict)]
        cmd += ["--verbose", str(verbose)]
        for i in range(n_jobs):
            print(' '.join(cmd))
            subprocess.run(cmd, env={"CUDA_VISIBLE_DEVICES": str(i)})


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

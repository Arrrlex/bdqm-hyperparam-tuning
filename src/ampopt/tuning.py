from joblib import Parallel, delayed
from ampopt.study import get_or_create_study
from ampopt.utils import parse_params, is_login_node, read_params_from_env, num_gpus
from ampopt.train import mk_objective

import os
from typing import Any, Dict

def tune(
    n_jobs: int = 1,
    n_trials_per_job: int = 10,
    study_name: str = None,
    with_db: bool = False,
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

    if with_db and study_name is None:
        print("If running on DB, study_name must be provided")
        print("Aborting")
        return

    if (not with_db) and n_jobs > 1:
        print("If running more than 1 job, with_db must be true")
        print("Aborting")
        return

    if 0 < num_gpus() < n_jobs:
        print(f"Warning: running {n_jobs} jobs with only {num_gpus()} GPUs, trouble ahead")

    local = "on DB" if with_db else "locally"
    print(f"Running hyperparam tuning {local} with:")
    print(f" - study_name: {study_name}")
    print(f" - dataset: {data}")
    print(f" - n_trials_per_job: {n_trials_per_job}")
    print(f" - sampler: {sampler}")
    print(f" - pruner: {pruner}")
    print(f" - num epochs: {n_epochs}")

    _ = get_or_create_study(
        study_name=study_name, with_db=with_db, pruner=pruner, sampler=sampler
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

    if not with_db:
        tune_local(
            study_name=study_name,
            with_db=with_db,
            pruner=pruner,
            sampler=sampler,
            n_epochs=n_epochs,
            data=data,
            n_trials=n_trials_per_job,
            params_dict=params_dict,
            gpu_device=0,
            verbose=verbose)
    else:
        Parallel(n_jobs=n_jobs)(
            delayed(tune_local)(
                study_name=study_name,
                with_db=with_db,
                pruner=pruner,
                sampler=sampler,
                n_epochs=n_epochs,
                data=data,
                n_trials=n_trials_per_job,
                params_dict=params_dict,
                gpu_device=i,
                verbose=verbose)
            for i in range(n_jobs)
        )



def tune_local(study_name: str, with_db: bool, pruner: str, sampler: str, n_epochs: int, data: str, n_trials: int, params_dict: Dict[str, Any], gpu_device: int, verbose: bool):
    os.environ["GPU_VISIBLE_DEVICES"] = str(gpu_device)
    study = get_or_create_study(
        study_name=study_name, with_db=with_db, pruner=pruner, sampler=sampler
    )
    objective = mk_objective(verbose=verbose, epochs=n_epochs, train_fname=data, **params_dict)
    study.optimize(objective, n_trials=n_trials)
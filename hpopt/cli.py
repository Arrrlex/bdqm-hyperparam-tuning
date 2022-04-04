from typing import List, Union, Literal

import typer

app = typer.Typer()


@app.command()
def tune(
    n_trials: int = typer.Option(10, help="number of trials (num of models to train)"),
    study_name: str = typer.Option(
        None, help="name of the study (required if with_db=True)"
    ),
    with_db: bool = typer.Option(
        False, help="store trials on MySQL database, or locally to this job"
    ),
    pruner: str = typer.Option("Median", help="which pruning algorithm to use"),
    sampler: str = typer.Option("CmaEs", help="which sampling algorithm to use"),
    verbose: bool = typer.Option(
        False, help="Whether or not to log the per-epoch results"
    ),
    n_epochs: int = typer.Option(100, help="number of epochs for each trial"),
):
    """
    Run HP tuning on this node.

    This command reads specific values for hyperparameters from the environment, and
    all other hyperparameters are tuned according to the ranges set in train.py.

    To set a hyperparameter to a specific value, the environment variable must be set
    like this:

    ```
    export param_num_layers=5;
    ```

    The hyperparameter `num_layers` will then be set to 5 during the hyperparameter
    optimization.

    ## Pruners

    - Median waits for 10 trials, then prunes the trial if, after 10 epochs,
    the MAE is higher than the median of the previous trials

    - Hyperband uses a Multi-Armed Bandit approach to pruning trials

    - None doesn't prune trials

    ## Samplers

    - CmaEs uses the Lightweight Covariance Matrix Adaptation Evolution Strategy

    - TPE uses the Tree-Structured Parzen Estimator

    - Random uses a random search

    - Grid uses a grid search (note: the code in study.py must be modified in
    to change the search space for Grid search)
    """
    from hpopt.study import get_or_create_study
    from hpopt.train import mk_objective
    from hpopt.utils import is_login_node, read_params_from_env

    if is_login_node():
        print("Don't run tuning on the login node!")
        print("Aborting")
        return

    local = "on DB" if with_db else "locally"
    print(f"Running hyperparam tuning {local} with:")
    print(f" - study_name: {study_name}")
    print(f" - n_trials: {n_trials}")
    print(f" - sampler: {sampler}")
    print(f" - pruner: {pruner}")
    print(f" - num epochs: {n_epochs}")

    params_dict = read_params_from_env()
    if params_dict:
        print(f" - params:")
        for k, v in params_dict.items():
            print(f"   - {k}: {v}")

    study = get_or_create_study(
        study_name=study_name, with_db=with_db, pruner=pruner, sampler=sampler
    )
    objective = mk_objective(verbose=verbose, epochs=n_epochs, **params_dict)
    study.optimize(objective, n_trials=n_trials)


@app.command()
def create_lmdbs(
    train: str = "train.traj", valid: str = "valid.traj", test: str = "test.traj"
) -> None:
    """
    Precompute GMP features and save to LMDB.

    Writes to train.lmdb, valid.lmdb and test.lmdb in the data directory.
    """
    print(f"Creating lmdbs from files {train}, {valid}, {test}")
    from hpopt.preprocess import create_lmdbs

    create_lmdbs(train_fname=train, valid_fname=valid, test_fname=test)


@app.command()
def train_valid_split(
    train: str = typer.Option("oc20_3k_train.traj", help="the input dataset"),
    valid_split: float = typer.Option(
        0.1, help="proportion of dataset to split off for validation"
    ),
    train_out_fname: str = typer.Option(
        "train.traj", help="filename to write output train dataset to"
    ),
    valid_out_fname: str = typer.Option(
        "valid.traj", help="filename to write output valid dataset to"
    ),
) -> None:
    """Split the dataset into train & valid sets."""
    print(f"Splitting {train}:")
    print(f"  {(1-valid_split)*100:.1f}% into {train_out_fname}")
    print(f"  {valid_split*100:.1f}% into {valid_out_fname}")

    from hpopt.preprocess import create_validation_split

    create_validation_split(
        train_fname=train,
        valid_split=valid_split,
        train_out_fname=train_out_fname,
        valid_out_fname=valid_out_fname,
    )


@app.command()
def delete_study(study_names: List[str]):
    """
    Delete study from the MySQL DB.
    """
    from hpopt.jobs import ensure_mysql_running
    from hpopt.study import delete_study

    ensure_mysql_running()
    for name in study_names:
        delete_study(name)
        print(f"Deleted study {name}.")


@app.command()
def generate_report(study: str):
    """
    Generate report for given study.

    The report will be saved to the folder `report/{study}`.
    """
    from hpopt.jobs import ensure_mysql_running
    from hpopt.study import generate_report

    ensure_mysql_running()
    generate_report(study)


@app.command()
def view_all_studies():
    """
    View basic information about all studies in the DB.
    """
    from hpopt.jobs import ensure_mysql_running
    from hpopt.study import get_all_studies

    ensure_mysql_running()

    studies = get_all_studies()
    for study in studies:
        print(f"Study {study.study_name}:")
        print(f"  Params:")
        for param in study.best_trial.params:
            print(f"    - {param}")
        print(f"  Best score: {study.best_trial.value}")
        print(f"  Num trials: {study.n_trials}")


@app.command()
def run_tuning_jobs(
    study_name: str = typer.Option(..., help="name of the study"),
    n_jobs: int = typer.Option(5, help="Number of PACE jobs to run"),
    n_trials_per_job: int = typer.Option(
        10, help="number of trials (num of models to train)"
    ),
    pruner: str = typer.Option("Median", help="which pruning algorithm to use"),
    sampler: str = typer.Option("CmaEs", help="which sampling algorithm to use"),
    n_epochs: int = typer.Option(100, help="number of epochs for each trial"),
    params: str = typer.Option("", help="comma-separated list of key=value HP pairs"),
):
    """
    Run multiple hyperparameter tuning PACE jobs.

    If the study name already exists, this command will add extra trials to that DB.

    ## Pruners

    - Median waits for 10 trials, then prunes the trial if, after 10 epochs,
    the MAE is higher than the median of the previous trials

    - Hyperband uses a Multi-Armed Bandit approach to pruning trials

    - None doesn't prune trials

    ## Samplers

    - CmaEs uses the Lightweight Covariance Matrix Adaptation Evolution Strategy

    - TPE uses the Tree-Structured Parzen Estimator

    - Random uses a random search

    - Grid uses a grid search (note: the code in study.py must be modified in
    to change the search space for Grid search)
    """
    from hpopt.jobs import run_tuning_jobs

    run_tuning_jobs(
        n_jobs=n_jobs,
        n_trials_per_job=n_trials_per_job,
        study_name=study_name,
        pruner=pruner,
        sampler=sampler,
        params=params,
        n_epochs=n_epochs,
    )


@app.command()
def run_mysql():
    """
    If MySQL job is not running, start it.
    """
    from hpopt.jobs import ensure_mysql_running

    ensure_mysql_running()


@app.command()
def view_running_jobs(name: str = None):
    """
    View list of all running jobs for the current user.
    """
    from hpopt.jobs import get_running_jobs

    running_jobs = get_running_jobs(job_name=name)
    if len(running_jobs) == 0:
        if name is None:
            print("No running jobs.")
        else:
            print(f"No running jobs with name {name}.")
    else:
        cols = ["id", "username", "queue", "name", "time", "elapsed", "status", "node"]
        print(running_jobs[cols].set_index("id"))

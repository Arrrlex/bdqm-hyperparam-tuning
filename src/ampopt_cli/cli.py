from typing import List, Optional

import typer

app = typer.Typer()

# Preprocessing


@app.command()
def compute_gmp(
    train: str = typer.Argument(..., help=".traj file to fit & compute features for"),
    others: Optional[List[str]] = typer.Argument(
        None, help="other .traj files to compute features for"
    ),
    data_dir: Optional[str] = typer.Option(
        "data", help="directory to write LMDB files into"
    ),
) -> None:
    """
    Precompute GMP features and save to LMDB.

    The LMDB files are written to the `data_dir` directory.

    Note: only the first argument TRAIN is used to fit the feature pipeline.
    """
    if others is None:
        others = []

    from ampopt.preprocess import compute_gmp

    compute_gmp(train, *others, data_dir=data_dir)


# Tuning


@app.command()
def tune(
    n_jobs: int = typer.Option(1, help="number of jobs to run in parallel"),
    n_trials_per_job: int = typer.Option(
        10, help="number of trials (num of models to train) per job"
    ),
    study_name: str = typer.Option(
        None, help="name of the study (required if with_db=True)"
    ),
    with_db: bool = typer.Option(
        False, help="store trials on MySQL database, or locally to this job"
    ),
    data: str = typer.Option("data/oc20_3k_train.lmdb", help="Train dataset"),
    pruner: str = typer.Option("Median", help="which pruning algorithm to use"),
    sampler: str = typer.Option("CmaEs", help="which sampling algorithm to use"),
    verbose: bool = typer.Option(
        False, help="Whether or not to log the per-epoch results"
    ),
    n_epochs: int = typer.Option(100, help="number of epochs for each trial"),
    params: str = typer.Option("", help="comma-separated list of key=value HP pairs"),
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
    from ampopt.tuning import tune

    tune(
        n_jobs=n_jobs,
        n_trials_per_job=n_trials_per_job,
        study_name=study_name,
        with_db=with_db,
        data=data,
        pruner=pruner,
        sampler=sampler,
        verbose=verbose,
        n_epochs=n_epochs,
        params=params,
    )


@app.command()
def run_pace_tuning_job(
    study_name: str = typer.Option(..., help="name of the study"),
    data: str = typer.Option("data/oc20_3k_train.lmdb", help="Train dataset"),
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
    Run hyperparameter tuning as a PACE job.

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
    from ampopt.jobs import run_pace_tuning_job

    run_pace_tuning_job(
        study_name=study_name,
        data=data,
        n_jobs=n_jobs,
        n_trials_per_job=n_trials_per_job,
        pruner=pruner,
        sampler=sampler,
        params=params,
        n_epochs=n_epochs,
    )


# Utilities


@app.command()
def generate_report(study: str):
    """
    Generate report for given study.

    The report will be saved to the folder `report/{study}`.
    """
    from ampopt.study import generate_report

    generate_report(study)


@app.command()
def delete_studies(study_names: List[str]):
    """
    Delete studies from the MySQL DB.
    """
    from ampopt.study import delete_studies

    delete_studies(*study_names)


@app.command()
def view_all_studies():
    """
    View basic information about all studies in the DB.
    """
    from ampopt.study import view_all_studies

    view_all_studies()


@app.command()
def ensure_mysql_running():
    """
    If MySQL job is not running, start it.
    """
    from ampopt.jobs import ensure_mysql_running

    ensure_mysql_running()


@app.command()
def show_running_jobs(name: str = None):
    """
    View list of all running jobs for the current user.
    """
    from ampopt.jobs import show_running_jobs

    show_running_jobs(name)

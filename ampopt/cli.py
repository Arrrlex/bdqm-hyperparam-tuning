from typing import List

import typer

app = typer.Typer()

# Preprocessing

@app.command()
def create_valid_split(
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
    from ampopt.preprocess import create_valid_split

    create_valid_split(
        train=train,
        valid_split=valid_split,
        train_out_fname=train_out_fname,
        valid_out_fname=valid_out_fname,
    )


@app.command()
def create_lmdbs(
    train: str = "train.traj", valid: str = "valid.traj", test: str = "test.traj"
) -> None:
    """
    Precompute GMP features and save to LMDB.

    Writes to train.lmdb, valid.lmdb and test.lmdb in the data directory.
    """
    from ampopt.preprocess import create_lmdbs

    create_lmdbs(train=train, valid=valid, test=test)

# Tuning

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
    from ampopt.train import tune

    tune(
        n_trials=n_trials,
        study_name=study_name,
        with_db=with_db,
        pruner=pruner,
        sampler=sampler,
        verbose=verbose,
        n_epochs=n_epochs,
    )


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
    from ampopt.jobs import run_tuning_jobs

    run_tuning_jobs(
        n_jobs=n_jobs,
        n_trials_per_job=n_trials_per_job,
        study_name=study_name,
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

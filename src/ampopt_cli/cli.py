from typing import List, Optional

import typer

app = typer.Typer()

# Preprocessing


@app.command()
def preprocess(
    train: str = typer.Argument(..., help=".traj file to fit & compute features for"),
    others: Optional[List[str]] = typer.Argument(
        None, help="other .traj files to compute features for"
    ),
    data_dir: Optional[str] = typer.Option(
        None, help="directory to write LMDB files into"
    ),
) -> None:
    """
    Scale, Precompute GMP features and save to LMDB.

    The LMDB files are written to the `data_dir` directory. By default, this is the
    `data` directory in the project root.

    Note: only the first argument TRAIN is used to fit the feature pipeline.
    """
    if others is None:
        others = []

    from ampopt import preprocess

    preprocess(train, *others, data_dir=data_dir)


# Tuning


@app.command()
def tune(
    jobs: int = typer.Option(1, help="number of jobs to run in parallel"),
    trials: int = typer.Option(
        ..., help="number of trials (num of models to train) per job"
    ),
    study: str = typer.Option(..., help="name of the study"),
    data: str = typer.Option(..., help="Train dataset"),
    pruner: str = typer.Option("Median", help="which pruning algorithm to use"),
    sampler: str = typer.Option("CmaEs", help="which sampling algorithm to use"),
    verbose: bool = typer.Option(
        False, help="Whether or not to log the per-epoch results"
    ),
    epochs: int = typer.Option(100, help="number of epochs for each trial"),
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
    from ampopt import tune

    tune(
        jobs=jobs,
        trials=trials,
        study=study,
        data=data,
        pruner=pruner,
        sampler=sampler,
        verbose=verbose,
        epochs=epochs,
        params=params,
    )


@app.command()
def run_pace_tuning_job(
    study: str = typer.Option(..., help="name of the study"),
    data: str = typer.Option(..., help="Train dataset"),
    trials: int = typer.Option(..., help="number of trials (num of models to train)"),
    pruner: str = typer.Option("Median", help="which pruning algorithm to use"),
    sampler: str = typer.Option("CmaEs", help="which sampling algorithm to use"),
    epochs: int = typer.Option(100, help="number of epochs for each trial"),
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
    from ampopt import run_pace_tuning_job

    run_pace_tuning_job(
        study=study,
        data=data,
        trials=trials,
        pruner=pruner,
        sampler=sampler,
        params=params,
        epochs=epochs,
    )


@app.command()
def tune_local(
    study: str = typer.Option(...),
    data: str = typer.Option(...),
    trials: int = typer.Option(...),
    epochs: int = typer.Option(...),
    params: str = typer.Option(""),
    verbose: bool = typer.Option(...),
):
    """For internal use only."""
    from ampopt.tuning import tune_local
    from ampopt.utils import parse_params

    tune_local(
        study_name=study,
        data=data,
        n_trials=trials,
        n_epochs=epochs,
        params_dict=parse_params(params),
        verbose=verbose,
    )


# Utilities


@app.command()
def generate_report(study: str):
    """
    Generate report for given study.

    The report will be saved to the folder `report/{study}`.
    """
    from ampopt import generate_report

    generate_report(study)


@app.command()
def delete_studies(studies: List[str]):
    """
    Delete studies from the MySQL DB.
    """
    from ampopt import delete_studies

    delete_studies(*studies)


@app.command()
def view_studies():
    """
    View basic information about all studies in the DB.
    """
    from ampopt import view_studies

    view_studies()


@app.command()
def ensure_mysql_running():
    """
    If MySQL job is not running, start it.
    """
    from ampopt import ensure_mysql_running

    ensure_mysql_running()


@app.command()
def view_jobs(name: str = None):
    """
    View list of all running jobs for the current user.
    """
    from ampopt import view_jobs

    view_jobs(name)

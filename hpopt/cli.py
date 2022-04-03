from typing import Optional, List

import typer

app = typer.Typer()


@app.command()
def tune(
    n_trials: int = 10,
    study_name: str = "distributed-amptorch-tuning",
    with_db: bool = False,
    pruner: str = "Median",
    sampler: str = "CmaEs",
    verbose: bool = False,
    n_epochs: int = 100,
):
    """
    Run HP tuning on this node.
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
        for k,v in params_dict.items():
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
    print(f"Creating lmdbs from files {train}, {valid}, {test}")
    from hpopt.preprocess import create_lmdbs

    create_lmdbs(train_fname=train, valid_fname=valid, test_fname=test)


@app.command()
def train_valid_split(
    train: str = "oc20_3k_train.traj",
    valid_split: float = 0.1,
    train_out_fname: str = "train.traj",
    valid_out_fname: str = "valid.traj",
) -> None:
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
    from hpopt.jobs import ensure_mysql_running
    from hpopt.study import delete_study

    ensure_mysql_running()
    for name in study_names:
        delete_study(name)
        print(f"Deleted study {name}.")


@app.command()
def generate_report(name: str):
    from hpopt.jobs import ensure_mysql_running
    from hpopt.study import generate_report

    ensure_mysql_running()
    generate_report(name)


@app.command()
def view_all_studies():
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
    n_jobs: int = 5,
    n_trials_per_job: int = 5,
    study_name: str = "distributed-amptorch-tuning",
    pruner: str="Median",
    sampler: str="CmaEs",
    n_epochs: int = 100,
    params: str = ""
):
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
    from hpopt.jobs import ensure_mysql_running

    ensure_mysql_running()


@app.command()
def view_running_jobs(name: str = None):
    from hpopt.jobs import get_running_jobs

    running_jobs = get_running_jobs(job_name=name)
    if len(running_jobs) == 0:
        if name is None:
            print("No running jobs.")
        else:
            print(f"No running jobs with name {name}.")
    else:
        cols = [
            "id", "username", "queue", "name", "time", "elapsed", "status", "node"
        ]
        print(running_jobs[cols].set_index("id"))

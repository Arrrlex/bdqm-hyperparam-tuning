from functools import lru_cache

import optuna
from optuna import visualization as viz
from optuna.samplers import TPESampler, CmaEsSampler, GridSampler, RandomSampler
from optuna.pruners import HyperbandPruner, MedianPruner, NopPruner
from dotenv import dotenv_values

from ampopt.utils import ampopt_path


@lru_cache
def connection_string() -> str:
    """Construct DB connection string from .env file."""
    config = dotenv_values(ampopt_path / ".env")
    username = config["MYSQL_USERNAME"]
    password = config["MYSQL_PASSWORD"]
    node = config["MYSQL_NODE"]
    db = config["HPOPT_DB"]
    return f"mysql+pymysql://{username}:{password}@{node}/{db}"


def delete_study(study_name: str):
    optuna.delete_study(study_name=study_name, storage=connection_string())
    print(f"Deleted study {study_name}.")


def delete_studies(*study_names: str):
    for study_name in study_names:
        delete_study(study_name)


def get_study(study_name: str):
    return optuna.load_study(study_name=study_name, storage=connection_string())


def get_all_studies():
    return optuna.get_all_study_summaries(storage=connection_string())


def get_or_create_study(study_name: str, sampler: str, pruner: str):
    samplers = {
        "CmaEs": CmaEsSampler(n_startup_trials=10),
        "TPE": TPESampler(n_startup_trials=40),
        "Random": RandomSampler(),
        "Grid": GridSampler(search_space={"num_layers": range(3, 9), "num_nodes": range(4, 16)}),
    }

    pruners = {
        "Hyperband": HyperbandPruner(),
        "Median": MedianPruner(n_startup_trials=10, n_warmup_steps=10),
        "None": NopPruner(),
    }

    return optuna.create_study(
        sampler=samplers[sampler],
        pruner=pruners[pruner],
        study_name=study_name,
        storage=connection_string(),
        load_if_exists=True,
    )


def view_studies():
    studies = get_all_studies()
    for study in studies:
        print(f"Study {study.study_name}:")
        try:
            trial = study.best_trial
            assert trial is not None
            print(f"  Params:")
            for param in trial.params:
                print(f"    - {param}")
            print(f"  Best score: {trial.value}")
        except AssertionError:
            print("  (no successful trials yet)")
        print(f"  Num trials: {study.n_trials}")


def generate_report(study_name: str):
    report_dir = ampopt_path / "report" / study_name
    try:
        report_dir.mkdir(parents=True)
    except FileExistsError:
        print(f"Report directory {report_dir} already exists.")
        return

    study = get_study(study_name)

    viz.plot_contour(study, params=["num_layers", "num_nodes"]).write_image(
        report_dir / "contour_plot.png")

    viz.plot_intermediate_values(study).write_image(
        report_dir / "intermediate.png"
    )

    viz.plot_optimization_history(study).write_image(
        report_dir / "history.png"
    )

    viz.plot_param_importances(study).write_image(
        report_dir / "param_importance.png"
    )

    print(f"Best params: {study.best_params} with MAE {study.best_value}")
    print(f"Report saved to {report_dir}")

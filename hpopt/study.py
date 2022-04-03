import optuna
from dotenv import dotenv_values

from hpopt.utils import bdqm_hpopt_path


def _construct_connection_string() -> str:
    """Construct DB connection string from .env file."""
    config = dotenv_values(bdqm_hpopt_path / ".env")
    username = config["MYSQL_USERNAME"]
    password = config["MYSQL_PASSWORD"]
    node = config["MYSQL_NODE"]
    db = config["HPOPT_DB"]
    return f"mysql+pymysql://{username}:{password}@{node}/{db}"


CONN_STRING = _construct_connection_string()


def delete_study(study_name: str):
    optuna.delete_study(study_name=study_name, storage=CONN_STRING)


def get_study(study_name: str):
    return optuna.load_study(study_name=study_name, storage=CONN_STRING)


def get_all_studies():
    return optuna.get_all_study_summaries(storage=CONN_STRING)


def get_or_create_study(study_name: str, with_db: str, sampler: str, pruner: str):
    samplers = {
        "CmaEs": optuna.samplers.CmaEsSampler(n_startup_trials=10),
        "TPE": optuna.samplers.TPESampler(n_startup_trials=40),
        "Random": optuna.samplers.RandomSampler(),
        "Grid": optuna.samplers.GridSampler(search_space={"num_layers": range(3, 9), "num_nodes": range(4, 16)}),
    }

    pruners = {
        "Hyperband": optuna.pruners.HyperbandPruner(),
        "Median": optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=10),
        "None": optuna.pruners.NopPruner(),
    }

    params = {
        "sampler": samplers[sampler],
        "pruner": pruners[pruner],
        "study_name": study_name,
    }

    if with_db:
        params["storage"] = CONN_STRING
        params["load_if_exists"] = True

    return optuna.create_study(**params)


def generate_report(study_name: str):
    report_dir = bdqm_hpopt_path / "report" / study_name
    try:
        report_dir.mkdir(parents=True)
    except FileExistsError:
        print(f"Report directory {report_dir} already exists.")
        return

    study = get_study(study_name)
    fig = optuna.visualization.plot_contour(study, params=["num_layers", "num_nodes"])
    fig.write_image(report_dir / "contour_plot.png")

    optuna.visualization.plot_intermediate_values(study).write_image(
        report_dir / "intermediate.png"
    )

    print(f"Best params: {study.best_params} with MAE {study.best_value}")
    print(f"Report saved to {report_dir}")

import optuna
from dotenv import dotenv_values
from utils import bdqm_hpopt_path


def _construct_connection_string() -> str:
    """Construct DB connection string from .env file"""
    config = dotenv_values(bdqm_hpopt_path / ".env")
    username = config["MYSQL_USERNAME"]
    password = config["MYSQL_PASSWORD"]
    node = config["MYSQL_NODE"]
    db = config["HPOPT_DB"]
    return f"mysql+pymysql://{username}:{password}@{node}/{db}"


CONN_STRING = _construct_connection_string()
STUDY_NAME = "distributed-amptorch-tuning"


def delete(study_name=STUDY_NAME):
    optuna.delete_study(study_name=study_name, storage=CONN_STRING)


def get_or_create(study_name=STUDY_NAME, with_db=True):
    params = {
        "pruner": optuna.pruners.HyperbandPruner(),
        "sampler": optuna.samplers.TPESampler(n_startup_trials=10),
        "study_name": study_name,
    }

    if with_db:
        params["storage"] = CONN_STRING
        params["load_if_exists"] = True

    return optuna.create_study(**params)


def get_best_params(study_name=STUDY_NAME):
    study = get_or_create(study_name=study_name, with_db=True)
    return f"Best params: {study.best_params} with MAE {study.best_value}"


def generate_report(study_name=STUDY_NAME):
    study = get_or_create(study_name=study_name, with_db=True)
    fig = optuna.visualization.plot_contour(study, params=["num_layers", "num_nodes"])
    fig.write_image("contour_plot.png")


if __name__ == "__main__":
    import sys

    cmd = sys.argv[1]
    if cmd == "delete":
        delete()
    elif cmd == "best-params":
        print(get_best_params())
    elif cmd == "report":
        generate_report()
    else:
        print(f'Command "{cmd}" not recognised')

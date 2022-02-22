import os

import optuna


def construct_connection_string():
    username = os.getenv("MYSQL_USERNAME")
    password = os.getenv("MYSQL_PASSWORD")
    node = os.getenv("MYSQL_NODE")
    db = os.getenv("DB")
    return f"mysql+pymysql://{username}:{password}@{node}/{db}"


if __name__ == "__main__":
    connection_string = construct_connection_string()
    optuna.delete_study(
        study_name="distributed-amptorch-tuning", storage=connection_string
    )

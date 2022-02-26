import os

import optuna

from utils import connection_string

if __name__ == "__main__":
    optuna.delete_study(
        study_name="distributed-amptorch-tuning", storage=connection_string
    )

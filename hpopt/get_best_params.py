import sys
import torch
from ase.io import Trajectory
from amptorch.trainer import AtomsTrainer
import os
import sys

import numpy as np
import optuna

from utils import connection_string

if __name__ == "__main__":
    study = optuna.load_study(
      study_name="distributed-amptorch-tuning", 
      storage=connection_string,
    )
    
    print(f"Best params: {study.best_params} with MAE {study.best_value}")

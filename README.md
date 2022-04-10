# AmpOpt
Code for Hyperparameter Optimization project for "Big Data &amp; Quantum
Mechanics" (Andrew Medford).

## Structure

- `jobs` contains `.pbs` files defining jobs to run on PACE
- `ampopt` contains code for preprocessing, running hyperparameter optimization
  jobs, and analyzing results
- `data` contains the source `.traj` files, as well as the preprocessed `.lmdb`
  files (once `ampopt create-lmdb` has been run).
- `env_gpu.yml` and `env_cpu.yml` contain dependencies for machines with and
  without (respectively) GPU.
- `setup-session.sh` is a convenience script for PACE jobs
- `setup.py` defines how `ampopt` Python package and command are installed.

## Installation
For setup and installation instructions, please refer to [SETUP.md](docs/SETUP.md).

## Usage

For usage instructions, please refer to [USAGE.md](docs/USAGE.md).

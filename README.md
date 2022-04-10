# AmpOpt
Code for Hyperparameter Optimization project for "Big Data &amp; Quantum
Mechanics" (Andrew Medford).

## Structure

- `jobs` contains `.pbs` files defining jobs to run on PACE
- `ampopt` contains code for preprocessing, running hyperparameter optimization
  jobs, and analyzing results
- `data` contains the source `.traj` files, as well as the preprocessed `.lmdb`
  files (once `ampopt create-lmdb` has been run).

## Installation
For setup and installation instructions, please refer to [SETUP.md](docs/SETUP.md).

## Usage

### Running Parallel Hyperparameter Tuning Jobs

1. Activate VPN and SSH into login node
2. Run `cd ~/bdqm-hyperparam-tuning`
3. Initialize conda: `source setup-session.sh`
4. Run `ampopt run-tuning-jobs --n-jobs=5 --n-trials-per-job=10`. Note: before
  running the tuning jobs, this script will check that MySQL is running and will
  start a MySQL job if not.

  For more configuration options, run `ampopt run-tuning-jobs --help`.

### Other Tasks

Run `ampopt --help` to see other available commands.

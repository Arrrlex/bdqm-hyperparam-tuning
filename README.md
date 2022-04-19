# AmpOpt<a name="ampopt"></a>

## Contents<a name="contents"></a>

<!-- mdformat-toc start --slug=github --maxlevel=6 --minlevel=1 -->

- [AmpOpt](#ampopt)
  - [Contents](#contents)
  - [Introduction](#introduction)
  - [Structure](#structure)
  - [Installation](#installation)
  - [Usage](#usage)

<!-- mdformat-toc end -->

## Introduction<a name="introduction"></a>

Code for Hyperparameter Optimization project for "Big Data & Quantum
Mechanics" (Andrew Medford).

## Structure<a name="structure"></a>

- `jobs` contains `.pbs` files defining jobs to run on PACE
- `src/ampopt` contains code for preprocessing, running hyperparameter optimization
  jobs, and analyzing results
- `src/ampopt_cli` exposes the functionality of `ampopt` via a CLI
- `data` contains the source `.traj` files, as well as the preprocessed `.lmdb`
  files (once `ampopt preprocess` has been run).
- `env_gpu.yml` and `env_cpu.yml` contain dependencies for machines with and
  without (respectively) GPU.
- `setup-session.sh` is a convenience script for PACE jobs
- `setup.py` defines how `ampopt` Python package and command are installed.

## Installation<a name="installation"></a>

For setup and installation instructions, please refer to [SETUP.md](docs/SETUP.md).

## Usage<a name="usage"></a>

For notebook tutorials, see:
- [Tutorial PACE](notebooks/Tutorial%20PACE.ipynb)
- [Tutorial Non-PACE](notebooks/Tutorial%20Non-PACE.ipynb)

For more comprehensive (but drier) instructions, please refer to
[USAGE.md](docs/USAGE.md).

Finally, for full explanation of all options, see `ampopt --help` or
`ampopt <command> --help`.

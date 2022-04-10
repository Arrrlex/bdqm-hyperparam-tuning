# Usage

All functionality is exposed via both Python and Command-Line interfaces.

## Preprocessing Data

In order to run a hyperparameter optimization job, AmpOpt requires data to be
in LMDB format, having been featurized using the preferred fingerprinting scheme.

AmpOpt provides a convenience function `compute_gmp` for using the GMP
fingerprinting scheme. It can be used as follows:

```bash
ampopt compute-gmp data/oc20_3k_train.traj data/oc20_300_test.traj
```

or

```python
from ampopt import compute_gmp
compute_gmp("data/oc20_3k_train.traj", "data/oc20_300_test.traj")
```

`compute_gmp` can accept arbitrarily many filenames as arguments, and as few as
one. Only the first filename will be used to fit the feature & target scalers.

`compute_gmp` chooses the output filenames by stripping the suffix (`.traj`)
and replacing it with `.lmdb`.

By default, `compute_gmp` writes to the `data` directory. To write to a
different directory:

```bash
ampopt compute-gmp data/oc20_3k_train.traj --data-dir=some/other/dir
```

or

```python
from ampopt import compute_gmp
compute_gmp("data/oc20_3k_train.traj", data_dir="some/other/dir")
```


## Tuning Hyperparameters

AmpOpt provides the function `tune` as the main interface for tuning
hyperparameters.

As a simple example:

```bash
ampopt tune --n-trials-per-job=2 --data data/oc20_3k_train.lmdb
```

or

```python
from ampopt import tune

tune(n_trials_per_job=2, data="data/oc20_3k_train.lmdb")
```

This will run a single hyperparameter tuning job locally (i.e. in the same
process) with 2 trials.

### Fixing Parameters

By default, AmpOpt will perform a Bayesian search over the entire hyperparameter
search space. To fix particular hyperparameters to a particular value, pass
those parameters in the `params` option as follows:

```bash
ampopt tune --n-trials-per-job=2 --params="dropout_rate=0.5,lr=1e-2"
```

or

```python
from ampopt import tune
tune(n_trials_per_job=2, params="dropout_rate=0.5,lr=1e-2")
```

The full list of parameters that can be set this way can be found by reading
the source code for `train.py`.

If `params` is set to the special value `env`, then the hyperparameters will
be read from the environment by looking for environment variables which start
with `param_`.

### Running Parallel Jobs

To run multiple jobs in parallel, the option `n_jobs` can be used:

```bash
ampopt tune --n-jobs=5 --n-trials-per-job=10 --with-db --study-name=example
```

If `--n-jobs` is used, it must be paired with `--with-db` and `--study-name`.

This will run 5 parallel processes with `Joblib`.

### Other Options

To see a full list of options for `tune`, run `ampopt tune --help`.

### Tuning as a PACE job

To run hyperparameter tuning as a PACE job, run

```bash
ampopt run-pace-tuning-job --study-name=example
```

## Other Tasks

AmpOpt has several utility functions for generating reports and interacting with
Optuna studies and PACE jobs.

Run `ampopt --help` to see all available commands.

### Utilities for PACE Jobs


You can check a PACE job's progress by running:

```
ampopt show-running-jobs
```

Take a look at the `name` column. You should see:

- One job named `mysql` (That's where the database is running)
- One jobs named `tune-amptorch-hy` if you're running a HP tuning job
- One job named `interactive-gpu-` if you're running an interactive session

In the column `status`, you'll see `R` (running) or `Q` (queued).

If you don't see your jobs at all, they might have already finished. You can
see all your jobs, including finished ones, by running `qstat -u $USER`.

Once your tuning job is finished, take a look at the stderr log files - they
should have a name like `tune-amptorch-hyperparams.e123456` where `123456` will
be the job's ID. If the job had an error, you'll see a traceback in that file.

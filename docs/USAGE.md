# Usage

All functionality is exposed via both Python and Command-Line interfaces.

## Preprocessing Data

In order to run a hyperparameter optimization job, AmpOpt requires data to be
in LMDB format, having been featurized using the preferred fingerprinting scheme.

AmpOpt provides a convenience function `compute_gmp` for using the GMP
fingerprinting scheme. It can be used as follows:

```bash
ampopt compute-gmp oc20_3k_train.traj test.traj
```

or

```python
from ampopt import compute_gmp
compute_gmp("oc20_3k_train.traj", "test.traj")
```

`compute_gmp` can accept arbitrarily many filenames as arguments, and as few as
one. Only the first filename will be used to fit the feature & target scalers.

`compute_gmp` chooses the output filenames by stripping the suffix (`.traj`)
and replacing it with `.lmdb`.


## Tuning Hyperparameters

AmpOpt provides 2 functions for tuning hyperparameters: `tune` and
`run-tuning-jobs`. `tune` is for running a job in the current process.
`run-tuning-jobs` queues several PACE jobs to run in parallel.

As an example:

```bash
ampopt run-tuning-jobs --study-name=example --n-jobs=5 --n-trials-per-job=10
```

or

```python
from ampopt import run_tuning_jobs
run_tuning_jobs(study_name="example", n_jobs=5, n_trials_per_job=10)
```

This will queue 5 PACE jobs, each of which will run 10 trials (i.e. train 10
models), for a total of 50 trials.

The only required argument is the study name, here "example".

The two functions share many parameters. To read about them, run

```bash
ampopt tune --help # or amptorch run-tuning-jobs --help
```

### Other Tasks

AmpOpt has several utility functions for generating reports and interacting with
Optuna studies and PACE jobs.

Run `ampopt --help` to see all available commands.
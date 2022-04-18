# AmpOpt Usage<a name="ampopt-usage"></a>

## Contents<a name="contents"></a>

<!-- mdformat-toc start --slug=github --maxlevel=6 --minlevel=1 -->

- [AmpOpt Usage](#ampopt-usage)
  - [Contents](#contents)
  - [Introduction](#introduction)
  - [Preprocessing Data](#preprocessing-data)
  - [Tuning Hyperparameters](#tuning-hyperparameters)
    - [Fixing Parameters](#fixing-parameters)
    - [Running Parallel Jobs](#running-parallel-jobs)
    - [Other Options](#other-options)
    - [Tuning as a PACE job](#tuning-as-a-pace-job)
  - [Other Tasks](#other-tasks)
    - [Utilities for PACE Jobs](#utilities-for-pace-jobs)
  - [Running A Single Trial](#running-a-single-trial)

<!-- mdformat-toc end -->

## Introduction<a name="introduction"></a>

All(ish) functionality is exposed via both Python and Command-Line interfaces.

(I haven't yet exposed functionality for running a single trial over cmdline).

## Preprocessing Data<a name="preprocessing-data"></a>

In order to run a hyperparameter optimization job, AmpOpt requires data to be
in LMDB format, having been featurized using the preferred fingerprinting scheme.

AmpOpt provides a convenience function `compute_gmp` for using the GMP
fingerprinting scheme. It can be used as follows:

```bash
ampopt compute-gmp data/oc20_3k_train.traj data/oc20_300_test.traj
```

or

```python
from ampopt.preprocess import compute_gmp
compute_gmp("data/oc20_3k_train.traj", "data/oc20_300_test.traj")
```

`compute_gmp` can accept arbitrarily many filenames as arguments, and as few as
one. Only the first filename will be used to fit the feature & target scalers.

File paths are specified relative to the current working directory.

`compute_gmp` chooses the output filenames by stripping the suffix (`.traj`)
and replacing it with `.lmdb`. For example, the above command will write files:

```
data/oc20_3k_train.lmdb
data/oc20_300_test.lmdb
```

By default, `compute_gmp` writes to the `data` directory in the project root. To
write to a different directory:

```bash
ampopt compute-gmp data/oc20_3k_train.traj --data-dir=some/other/dir
```

or

```python
from ampopt.preprocess import compute_gmp
compute_gmp("data/oc20_3k_train.traj", data_dir="some/other/dir")
```

This will create a file `some/other/dir/oc20_3k_train.lmdb`.

## Tuning Hyperparameters<a name="tuning-hyperparameters"></a>

AmpOpt provides the function `tune` as the main interface for tuning
hyperparameters.

As a simple example:

```bash
ampopt tune --study=example --trials=2 --data data/oc20_3k_train.lmdb
```

or

```python
from ampopt import tune

tune(study="example", trials=2, data="data/oc20_3k_train.lmdb")
```

This will run a single hyperparameter tuning job locally (i.e. in the same
process) with 2 trials, writing the results to the "example" study in the DB.

The data path is relative to the

### Fixing Parameters<a name="fixing-parameters"></a>

By default, AmpOpt will perform a Bayesian search over the entire hyperparameter
search space. To fix particular hyperparameters to a particular value, pass
those parameters in the `params` option as follows:

```bash
ampopt tune --study=fixed-params --trials=2 --data=data/oc20_3k_train.lmdb \
  --params="dropout_rate=0.5,lr=1e-2"
```

or

```python
from ampopt.tuning import tune
tune(
    study="fixed-params",
    trials=2,
    data="data/oc20_3k_train.lmdb",
    params="dropout_rate=0.5,lr=1e-2",
)
```

If you have the parameters as a dictionary, you can use the `format_params`
function from `ampopt.utils` as follows:

```python
from ampopt.tuning import tune
from ampopt.utils import format_params

params = {"dropout_rate": 0.5, "lr": 1e-2}
tune(
    study="fixed-params",
    trials=2,
    data="data/oc20_3k_train.lmdb",
    params=format_params(params),
)
```

The hyperparameters available are as follows (all ranges are inclusive):

- `num_layers`, the number of neural network layers. By default between 3 and 8
- `num_nodes`, the number of nodes in each layer. By default between 4 and 15
- `dropout_rate`. By default between 0 and 1
- `lr`, the learning rate. By default between 1e-5 and 0.1
- `gamma`, the rate at which the learning rate decays every 100 epochs. By
  default between 0.1 and 1

If `params` is set to the special value `env`, then the hyperparameters will
be read from the environment by looking for environment variables which start
with `param_`.

### Running Parallel Jobs<a name="running-parallel-jobs"></a>

To run multiple jobs in parallel, the option `jobs` can be used:

```bash
ampopt tune --jobs=5 --trials=10 --study=example-parallel --data=data/oc20_3k_train.lmdb
```

or

```python
from ampopt import tune
tune(trials=2, params="dropout_rate=0.5,lr=1e-2")
```

This will run 5 parallel processes with `subprocess`.

Note: to run parallel jobs on PACE, refer to the section
[Tuning as a PACE Job](#tuning-as-a-pace-job).

### Other Options<a name="other-options"></a>

To see a full list of options for `tune`, run `ampopt tune --help`.

### Tuning as a PACE job<a name="tuning-as-a-pace-job"></a>

To run hyperparameter tuning as a PACE job, run

```bash
ampopt run-pace-tuning-job --study=example-pace --trials=2 --data=data/oc20_3k_train.lmdb
```

or

```python
from ampopt.jobs import run_pace_tuning_job

run_pace_tuning_job(study="example-pace", trials=2, data="data/oc20_3k_train.lmdb")
```

Note: to run several tuning jobs in parallel, simply call this function multiple
times:

```python
for _ in range(5):
    run_pace_tuning_job(study="example-pace", trials=2, data="data/oc20_3k_train.lmdb")
```

## Other Tasks<a name="other-tasks"></a>

AmpOpt has several utility functions for generating reports and interacting with
Optuna studies and PACE jobs.

Run `ampopt --help` to see all available commands.

### Utilities for PACE Jobs<a name="utilities-for-pace-jobs"></a>

You can check a PACE job's progress by running:

```
ampopt view-jobs
```

or

```
from ampopt.jobs import view_jobs

view_jobs()
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

## Running A Single Trial<a name="running-a-single-trial"></a>

If you want to just run a single trial with given hyperparameters and see the
result, you can do so as follows:

```python
from ampopt.train import eval_score

print(eval_score(
    train_fname="data/oc20_3k_train.traj",
    epochs=100,
    num_layers=5,
    num_nodes=10,
    dropout_rate=0.,
    lr=1e-3,
    gamma=1,
))
```

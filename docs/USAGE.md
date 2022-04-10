# Usage

All functionality is exposed via both Python and Command-Line interfaces.

## Preprocessing Data

In order to run a hyperparameter optimization job, AmpOpt requires that data
be already split into train and valid sets, in LMDB format, having already been
featurized using the preferred fingerprinting scheme.

AmpOpt provides convenience functions to split data and perform GMP
fingerprinting.

### Train-Valid Splitting

To split a dataset into train and validate splits, the dataset must first be
placed in the `data` folder. AmpOpt expects the data to be loadable as a
trajectory object using `ase`. For this example, we will use the provided
`oc20_3k_train.traj` file.

Using the CLI, the command is:

```bash
ampopt create-valid-split oc20_3k_train.traj
```

Using Python, the code is:

```python
from ampopt import create_valid_split
create_valid_split("oc20_3k_train.traj")
```

This will create 2 files in the `data` folder:

- `train.traj` containing a random 90% of `oc20_3k_train.traj`, and
- `valid.traj` containing the remaining 10%

To write to different filenames, the options `train-out-fname` and
`valid-out-fname` can be provided:

```bash
ampopt create-valid-split oc20_3k_train.traj --train-out-fname=special_train.traj --valid-out-fname=special_test.traj
```

or

```python
from ampopt import create_valid_split
create_valid_split("oc20_3k_train.traj", train_out_fname="special_train.traj", valid_out_fname="special_test.traj")
```

To change the split %, the parameter `valid-split` can be provided with any
value between 0 and 1:

```bash
ampopt create-valid-split oc20_3k_train.traj --valid-split=0.4
```

or

```python
from ampopt import create_valid_split
create_valid_split("oc20_3k_train.traj", valid_split=0.4)
```

### Fingerprinting and Writing to LMDB

Once data has been split, it must be fingerprinted and written to LMDB format.
This can be done using the `create_lmdbs` function:

```bash
ampopt create-lmdbs
```

or

```python
from ampopt import create_lmdbs
create_lmdbs()
```

By default, the files `train.traj`, `valid.traj` and `test.traj` will be used.
To specify different file names:

```bash
ampopt create-lmdbs --train=train2.traj --test=test2.traj --valid=valid2.traj
```

or

```python
from ampopt import create_lmdbs
create_lmdbs(train="train2.traj", valid="valid2.traj", test="test2.traj")
```

`create_lmdbs` will write to the files `train.lmdb`, `valid.lmdb` and
`test.lmdb`. This is not configurable.


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
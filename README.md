# bdqm-hyperparam-tuning
Code for Hyperparameter Optimization project for "Big Data &amp; Quantum
Mechanics" (Andrew Medford).

## Structure

- `jobs` contains `.pbs` files defining jobs to run on PACE
- `hpopt` contains code for preprocessing, running hyperparameter optimization
  jobs, and analyzing results
- `data` contains the source `.traj` files, as well as the preprocessed `.lmdb`
  files (once `hpopt create-lmdb` has been run).

## Usage
### One-Time Setup on PACE-ICE

1. Activate the Gatech VPN (https://docs.pace.gatech.edu/gettingStarted/vpn/)
2. Log in to the login node:

    ```bash
    ssh <your-gatech-username>@pace-ice.pace.gatech.edu
    ```

3. Clone repos:

    ```bash
    git clone https://github.com/Arrrlex/amptorch.git
    git clone https://github.com/Arrrlex/bdqm-hyperparam-tuning.git
    ```

4. Start an interactive job:

    ```
    qsub ~/bdqm-hyperparam-tuning/jobs/interactive-gpu-session.pbs
    ```

5. Activate the conda module:

    ```
    module load anaconda3/2021.05
    ```

6. Create the conda environment and install the project into it:

    ```
    conda env create -f ~/bdqm-hyperparam-tuning/env_gpu.yml
    conda activate bdqm-hpopt
    pip install -e ~/bdqm-hyperparam-tuning
    ```

7. Switch to the right amptorch branch and install it into the conda env:

    ```
    cd ~/amptorch
    git checkout BDQM_VIP_2022Feb
    pip install -e .
    ```

8. Quit the job:

    ```
    exit
    ```

9. Install MySQL, following [these instructions](https://docs.pace.gatech.edu/software/mysql/) (in particular the “multi-node access” section)
10. Create a file `~/bdqm-hyperparam-tuning/.env` with the following contents:

    ```
    MYSQL_USERNAME=... # your gatech username
    MYSQL_PASSWORD=... # the mysql password you set in step 9
    HPOPT_DB=hpopt
    ```

### Running Code Interactively

1. Activate VPN
2. SSH into login node
3. Start an interactive job:

    ```
    qsub ~/bdqm-hyperparam-tuning/jobs/interactive-gpu-session.pbs
    ```

4. Set up the conda environment:

    ```
    source ~/bdqm-hyperparam-tuning/setup-session.sh
    ```

5. Run some code

### Running Parallel Hyperparameter Tuning Jobs

1. Activate VPN and SSH into login node
2. Run `cd ~/bdqm-hyperparam-tuning`
3. Initialize conda: `source setup-session.sh`
4. Run `hpopt run-tuning-jobs --n-jobs=5 --n-trials-per-job=10`. Note: before
  running the tuning jobs, this script will check that MySQL is running and will
  start a MySQL job if not.

  For more configuration options, run `hpopt run-tuning-jobs --help`.

### Other Tasks

Run `hpopt --help` to see other available commands.
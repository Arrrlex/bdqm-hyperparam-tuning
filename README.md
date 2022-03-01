# bdqm-hyperparam-tuning
Code for Hyperparameter Optimization project for "Big Data &amp; Quantum Mechanics" (Andrew Medford).

## Structure

- jobs
  - Contains code for running and configuring jobs on PACE-ICE
- hpopt
  - Contains code for running hyperparameter optimization jobs, as well as analysing the results

## One-Time Setup on PACE-ICE

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
    qsub ~/bdqm-hyperparam-tuning/interactive-gpu-session.pbs
    ```

5. Activate the conda module:

    ```
    module load anaconda3/2021.05
    ```

6. Create the conda environment:

    ```
    conda env create -f ~/bdqm-hyperparameter-tuning/env_gpu.yml
    conda activate bdqm-hpopt
    ```

7. Switch to the right amptorch branch and install it in the conda env:

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

## Running Code Interactively

1. Activate VPN
2. SSH into login node
3. Start an interactive job:

    ```
    qsub ~/bdqm-hyperparam-tuning/interactive-gpu-session.pbs
    ```

4. Set up the conda environment:

    ```
    source ~/bdqm-hyperparam-tuning/setup-session.sh
    ```

5. Run some code

## Running Parallel Hyperparameter Tuning Jobs

1. Activate VPN and SSH into login node
2. Run `cd ~/bdqm-hyperparam-tuning`
3. Initialize conda: `source jobs/setup-session.sh`
4. Run `./jobs/run-tuning-jobs.sh 5 50`. This will run 5 tuning jobs, each of which will run for 50 trials. This script will also check if a MySQL server is running, and start one if not.

## Other Scripts

- `hpopt/delete_study.py` can be used to delete the previous trials in the study
- `hpopt/create_validation_set.py` splits the data into train, validation, and test sets
- `hpopt/create_lmdb.py` Runs featurization on the train, validate, and test datasets, and writes them in lmdb format.
- `hpopt/generate_report.py` creates a report based on the tuning
- `hpopt/get_best_params.py` returns which parameters got the lowest score, and what score they attained.

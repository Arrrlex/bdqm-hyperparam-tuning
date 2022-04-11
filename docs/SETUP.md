# SETUP

This document will take you through setting up your system to run parallel
GPU-accelerated hyperparameter optimization jobs with `ampopt`.

If you're on the PACE cluster, follow the instructions below ("Setup on PACE").
Otherwise, skip ahead to the section "Setup on Generic System".

## Setup on PACE

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

    Note: all the following steps must happen inside the interactive job.

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
8. **Install MySQL**:
    1. If you've tried to install MySQL before, delete the old attempt: `rm -rf ~/scratch/db`
    2. Run:

        ```
        $ export DB_DIR=$HOME/scratch/db
        $ mkdir -p $DB_DIR

        $ cat << EOF > ~/.my.cnf
        [mysqld]
        datadir=$DB_DIR
        socket=$DB_DIR/mysqldb.sock
        user=$USER
        symbolic-links=0
        skip-networking

        [mysqld_safe]
        log-error=$DB_DIR/mysqldb.log
        pid-file=$DB_DIR/mysqldb.pid

        [mysql]
        socket=$DB_DIR/mysqldb.sock
        EOF
        ```
    3. Run `mysql_install_db --datadir=$DB_DIR`
    4. Run `mysqld_safe &`
    5. Choose a password and make a note of it
    5. Run this, **replacing 'my-secure-password' with the password you just
       chose**:

        ```
        mysql -u root << EOF
        UPDATE mysql.user SET Password=PASSWORD(RAND()) WHERE User='root';
        DELETE FROM mysql.user WHERE User='root' AND Host NOT IN ('localhost', '127.0.0.1', '::1');
        DELETE FROM mysql.user WHERE User='';
        DELETE FROM mysql.db WHERE Db='test' OR Db='test_%';
        GRANT ALL PRIVILEGES ON *.* TO '$USER'@'%' IDENTIFIED BY 'my-secure-password' WITH GRANT OPTION;
        FLUSH PRIVILEGES;
        EOF
        ```

    6. Run `mysql -u $USER -p`. When prompted, enter your MySQL password.
    7. At the MySQL prompt, run `CREATE DATABASE hpopt;`
    8. Exit the MySQL prompt, e.g. run `exit`
    9. Quit the interactive job, e.g. run `exit` again to get back to the login
       node.

9. Create a file `~/bdqm-hyperparam-tuning/.env` with the following contents:

    ```
    MYSQL_USERNAME=... # your gatech username
    MYSQL_PASSWORD=... # the mysql password you set in step 8
    HPOPT_DB=hpopt
    MYSQL_NODE=placeholder
    ```

## Setup on Generic System

1. Clone repos:

    ```bash
    git clone https://github.com/Arrrlex/amptorch.git
    git clone https://github.com/Arrrlex/bdqm-hyperparam-tuning.git
    ```
2. Ensure conda is installed
3. Change to the project directory:

    ```bash
    cd bdqm-hyperparam-tuning
    ```

3. Create the conda environment by running either `conda env create -f env_cpu.yml`
   or `conda env create -f env_gpu.yml` depending on if your system has a GPU
   available or not.

4. Install both packages locally into the conda environment:

    ```
    conda activate bdqm-hpopt
    pip install -e .
    cd ../amptorch
    pip install -e .
    ```

5. **Install MySQL**:
    1. Instructions vary depending on your platform, but on mac to install mysql
       you can just run `brew install mysql`.
    2. Start mysql by running `brew services start mysql` (again, this will
       be different for linux users).
    2. Choose a password and make a note of it
    3. Run this, **replacing 'my-secure-password' with the password you just chose**:

        ```
        mysqladmin -u root password 'my-secure-password'`
        ```

    4. Run `mysql -u root -p`. When prompted, enter your MySQL password.
    5. At the MySQL prompt, run `CREATE DATABASE hpopt;`
    6. Exit the MySQL prompt by running `exit`

6. Create a file `.env` in the `bdqm-hyperparam-tuning` folder with the
   following contents:

    ```
    MYSQL_USERNAME=root
    MYSQL_PASSWORD=... # the mysql password you set in step 8
    HPOPT_DB=hpopt
    MYSQL_NODE=localhost
    ```

## First Steps Running Hyperparameter Optimization Code

Follow these steps to verify that everything is setup correctly.

1. If on PACE:
    1. Activate VPN, SSH into login node
    2. Start an interactive job:

        ```
        qsub ~/bdqm-hyperparam-tuning/jobs/interactive-gpu-session.pbs
        ```

    3. Change to the project directory:

        ```
        cd bdqm-hyperparam-tuning
        ```

    3. Set up the conda environment:

        ```
        source setup-session.sh
        ```

    If not on PACE:

    1. Make sure that MySQL is running
    2. Change to the project directory:

        ```bash
        cd bdqm-hyperparam-tuning
        ```

    3. Activate the conda environment:

        ```bash
        conda activate bdqm-hpopt
        ```

2. First, we have to generate the LMDB files:

    ```
    ampopt compute-gmp data/oc20_3k_train.traj data/oc20_300_test.traj
    ```

3. Next, try running a small tuning job on the current node:

    ```
    ampopt tune --n-trials=1 --n-epochs=5
    ```

    If that worked, congrats! You managed to train a simple AmpTORCH model.

4. The next step is to run a tuning job backed by a DB:

    ```
    ampopt tune --n-trials=1 --n-epochs=5 --with-db
    ```
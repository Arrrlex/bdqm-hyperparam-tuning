# SETUP<a name="setup"></a>

## Contents<a name="contents"></a>

<!-- mdformat-toc start --slug=github --maxlevel=6 --minlevel=1 -->

- [SETUP](#setup)
  - [Contents](#contents)
  - [Introduction](#introduction)
  - [Setup on PACE](#setup-on-pace)
  - [Setup on Generic System](#setup-on-generic-system)
  - [First Steps Running Hyperparameter Optimization Code](#first-steps-running-hyperparameter-optimization-code)

<!-- mdformat-toc end -->

## Introduction<a name="introduction"></a>

This document will take you through setting up your system to run parallel
GPU-accelerated hyperparameter optimization jobs with `ampopt`.

If you're on the PACE cluster, follow the instructions below ("Setup on PACE").
Otherwise, skip ahead to the section "Setup on Generic System".

## Setup on PACE<a name="setup-on-pace"></a>

1. Activate the Gatech VPN (https://docs.pace.gatech.edu/gettingStarted/vpn/)

1. Log in to the login node:

   ```bash
   ssh <your-gatech-username>@pace-ice.pace.gatech.edu
   ```

1. Clone repos:

   ```bash
   git clone https://github.com/Arrrlex/amptorch.git
   git clone https://github.com/Arrrlex/bdqm-hyperparam-tuning.git
   ```

1. Start an interactive job:

   ```
   qsub ~/bdqm-hyperparam-tuning/jobs/interactive-gpu-session.pbs
   ```

   Note: all the following steps must happen inside the interactive job.

1. Activate the conda module:

   ```
   module load anaconda3/2021.05
   ```

1. Create the conda environment and install the project into it:

   ```
   conda env create -f ~/bdqm-hyperparam-tuning/env_gpu.yml
   conda activate bdqm-hpopt
   pip install -e ~/bdqm-hyperparam-tuning
   ```

1. Switch to the right amptorch branch and install it into the conda env:

   ```
   cd ~/amptorch
   git checkout BDQM_VIP_2022Feb
   pip install -e .
   ```

1. **Install MySQL**:

   1. If you've tried to install MySQL before, delete the old attempt: `rm -rf ~/scratch/db`

   1. Run:

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

   1. Run `mysql_install_db --datadir=$DB_DIR`

   1. Run `mysqld_safe &`

   1. Choose a password and make a note of it

   1. Run this, **replacing 'my-secure-password' with the password you just
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

   1. Run `mysql -u $USER -p`. When prompted, enter your MySQL password.

   1. At the MySQL prompt, run `CREATE DATABASE hpopt;`

   1. Exit the MySQL prompt, e.g. run `exit`

   1. Quit the interactive job, e.g. run `exit` again to get back to the login
      node.

1. Create a file `~/bdqm-hyperparam-tuning/.env` with the following contents:

   ```
   MYSQL_USERNAME=... # your gatech username
   MYSQL_PASSWORD=... # the mysql password you set in step 8
   HPOPT_DB=hpopt
   MYSQL_NODE=placeholder
   ```

## Setup on Generic System<a name="setup-on-generic-system"></a>

1. Clone repos:

   ```bash
   git clone https://github.com/Arrrlex/amptorch.git
   git clone https://github.com/Arrrlex/bdqm-hyperparam-tuning.git
   ```

1. Ensure conda is installed

1. Change to the project directory:

   ```bash
   cd bdqm-hyperparam-tuning
   ```

1. Create the conda environment by running either `conda env create -f env_cpu.yml`
   or `conda env create -f env_gpu.yml` depending on if your system has a GPU
   available or not.

1. Install both packages locally into the conda environment:

   ```
   conda activate bdqm-hpopt
   pip install -e .
   cd ../amptorch
   pip install -e .
   ```

1. **Install MySQL**:

   1. Instructions vary depending on your platform, but on mac to install mysql
      you can just run `brew install mysql`.

   1. Start mysql by running `brew services start mysql` (again, this will
      be different for linux users).

   1. Choose a password and make a note of it

   1. Run this, **replacing 'my-secure-password' with the password you just chose**:

      ```
      mysqladmin -u root password 'my-secure-password'`
      ```

   1. Run `mysql -u root -p`. When prompted, enter your MySQL password.

   1. At the MySQL prompt, run `CREATE DATABASE hpopt;`

   1. Exit the MySQL prompt by running `exit`

1. Create a file `.env` in the `bdqm-hyperparam-tuning` folder with the
   following contents:

   ```
   MYSQL_USERNAME=root
   MYSQL_PASSWORD=... # the mysql password you set in step 8
   HPOPT_DB=hpopt
   MYSQL_NODE=localhost
   ```
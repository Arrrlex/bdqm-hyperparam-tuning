#!/usr/bin/env bash

NUM_JOBS=$1
NUM_TRIALS=$1

echo $NUM_TRIALS > ~/bdqm-hyperparam-tuning/.num_trials

source jobs/setup-session.sh
python jobs/start_mysql_server.py

for i in $(seq 1 $1)
do
  qsub ~/bdqm-hyperparam-tuning/jobs/submit-hpopt-job.pbs
done

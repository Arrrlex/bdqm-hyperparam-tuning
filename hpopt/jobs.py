import sys
import time
from pathlib import Path
import subprocess
import re

import pandas as pd

from hpopt.utils import bdqm_hpopt_path


def to_path(job_name):
    return bdqm_hpopt_path / f"jobs/{job_name}.pbs"

def check_job_valid(job_name):
    config = to_path(job_name).read_text()
    match = re.search(r"^#PBS -N (.*?)$", config, re.M)
    assert match.group(1).strip() == job_name

def queue_job(job_name, **extra_args):
    path = to_path(job_name)
    extras = ",".join(f"{k}={v}" for k,v in extra_args.items())
    cmd = f"qsub {path}"
    if extras:
        cmd += f" -v \"{extras}\""
    subprocess.run(cmd, shell=True)

def qstat():
    qstat_result = subprocess.run("qstat -u $USER -n1", shell=True, capture_output=True)
    data = [r.split() for r in qstat_result.stdout.decode("utf-8").splitlines() if r and r[0].isnumeric()]
    columns = ["id", "username", "queue", "name", "sessid", "nds", "tsk", "memory", "time", "status", "elapsed", "node"]
    df = pd.DataFrame(data, columns=columns)
    df["id"] = df["id"].apply(lambda x: x.split(".")[0])
    return df

def get_running_jobs(job_name: str = None):
    jobs = qstat().query("status.isin(['Q', 'R'])")
    if job_name is not None:
        jobs = jobs[jobs.name == job_name]
    return jobs

def get_or_start(job_name):
    check_job_valid(job_name)
    jobs = get_running_jobs(job_name)
    if len(jobs) == 1:
        job = jobs.iloc[0]
        if job.status == "Q":
            print(f"Waiting for {job_name} job {job.id} to start...")
            time.sleep(10)
            return get_or_start(job_name)
        print(f"{job_name} running, job ID: {job.id}")
        return job
    elif len(jobs) > 1:
        print(f"More than 1 {job_name} jobs running - aborting")
        sys.exit(1)
    else:
        print(f"Starting {job_name} job")
        queue_job(job_name)
        return get_or_start(job_name)

def update_dotenv_file(node):
    with open(bdqm_hpopt_path / ".env") as f:
        lines = f.readlines()

    with open(bdqm_hpopt_path / ".env", "w") as f:
        for line in lines:
            if not line.startswith("MYSQL_NODE"):
                f.write(line)
        f.write(f"MYSQL_NODE={node}")

def ensure_mysql_running():
    job = get_or_start("mysql")
    update_dotenv_file(job.node)


def run_tuning_jobs(n_jobs: int, n_trials_per_job: int, study_name: str, pruner: str, sampler: str):
    print(f"Running {n_jobs} tuning jobs with {n_trials_per_job} trials per job")
    print(f"Study_name: {study_name}, pruner: {pruner}, sampler: {sampler}")

    ensure_mysql_running()
    
    for i in range(n_jobs):
        queue_job("tune-amptorch-hyperparams", n_trials=n_trials_per_job, study_name=study_name, pruner=pruner, sampler=sampler)    

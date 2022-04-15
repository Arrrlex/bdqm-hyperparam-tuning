"""Contains code for scheduling and viewing PACE jobs."""

import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from ampopt.utils import ampopt_path, parse_params


def run_pace_tuning_job(
    study_name: str,
    data="data/oc20_3k_train.lmdb",
    n_jobs: int = 5,
    n_trials_per_job: int = 10,
    pruner: str = "Median",
    sampler: str = "CmaEs",
    params: str = "",
    n_epochs: int = 100,
):
    ensure_mysql_running()
    params_dict = parse_params(params, prefix="param_")

    queue_job(
        "tune-amptorch-hyperparams",
        template_args={"n_jobs": n_jobs},
        n_jobs=n_jobs,
        n_trials=n_trials_per_job,
        data=data,
        study_name=study_name,
        pruner=pruner,
        sampler=sampler,
        n_epochs=n_epochs,
        **params_dict,
    )


def to_path(job_name: str) -> Path:
    """Convert a job name to a filepath."""
    return ampopt_path / f"jobs/{job_name}.pbs"


def check_job_valid(job_name: str):
    """
    Raise AssertionError if either:
     - job_name doesn't exist, or
     - job_name's .pbs file has a different job name
    """
    config = to_path(job_name).read_text()
    match = re.search(r"^#PBS -N (.*?)$", config, re.M)
    assert match.group(1).strip() == job_name


def queue_job(job_name, template_args=None, **extra_args):
    """
    Schedule a job to be run.

    template_args are used to fill in the job template.
    **extra_args are passed as environment variables to the job script.
    """
    path = to_path(job_name)
    if template_args is not None:
        path = apply_template_args(path, template_args)
    extras = ",".join(f"{k}={v}" for k, v in extra_args.items())
    cmd = f"qsub {path}"
    if extras:
        cmd += f' -v "{extras}"'
    subprocess.run(cmd, shell=True)


def apply_template_args(path, template_args):
    pbs_script = path.read_text().format(**template_args)
    stem = path.stem + "_" + "_".join(f"{k}_{v}" for k, v in sorted(template_args.items()))
    new_path = path.parent / f"{stem}.pbs"
    new_path.write_text(pbs_script)
    return new_path


def qstat():
    """
    Return the result of the command `qstat -u $USER -n1` command as a pandas DataFrame.

    This command returns a list of all the current user's jobs.
    """
    qstat_result = subprocess.run("qstat -u $USER -n1", shell=True, capture_output=True)
    data = [
        r.split()
        for r in qstat_result.stdout.decode("utf-8").splitlines()
        if r and r[0].isnumeric()
    ]
    columns = [
        "id",
        "username",
        "queue",
        "name",
        "sessid",
        "nds",
        "tsk",
        "memory",
        "time",
        "status",
        "elapsed",
        "node",
    ]
    df = pd.DataFrame(data, columns=columns)
    df["id"] = df["id"].apply(lambda x: x.split(".")[0])
    return df


def get_running_jobs(job_name: str = None):
    """
    Return pandas DataFrame consisting of current user's running or queued jobs whose
    name is `job_name`.

    If `job_name` is None, return all running jobs for current user.
    """
    jobs = qstat().query("status.isin(['Q', 'R'])")
    if job_name is not None:
        jobs = jobs[jobs.name == job_name]
    return jobs


def show_running_jobs(job_name: str = None):
    """
    Print current user's running jobs whose name is `job_name`.

    If `job_name` is None, print all running jobs for current user.
    """

    running_jobs = get_running_jobs(job_name=job_name)
    if len(running_jobs) == 0:
        if job_name is None:
            print("No running jobs.")
        else:
            print(f"No running jobs with name {job_name}.")
    else:
        print(running_jobs.to_string(index=False))


def get_or_start(job_name, just_queued=False):
    """
    Check if a job with name `job_name` is running, and if not queue it and wait for
    it to start.

    Return a pandas Series with information about the running job
    """
    check_job_valid(job_name)
    jobs = get_running_jobs(job_name)
    if len(jobs) == 1:
        job = jobs.iloc[0]
        if job.status == "Q":
            print(f"Waiting for {job_name} job {job.id} to start...")
            time.sleep(10)
            return get_or_start(job_name, just_queued=just_queued)
        print(f"{job_name} running, job ID: {job.id}")
        if just_queued:
            time.sleep(5)
        return job
    elif len(jobs) > 1:
        print(f"More than 1 {job_name} jobs running - aborting")
        sys.exit(1)
    else:
        print(f"Starting {job_name} job")
        queue_job(job_name)
        return get_or_start(job_name, just_queued=True)


def update_dotenv_file(node):
    """
    Update .env with information about the running MySQL job.
    """
    with open(ampopt_path / ".env") as f:
        lines = f.readlines()

    with open(ampopt_path / ".env", "w") as f:
        for line in lines:
            if not line.startswith("MYSQL_NODE"):
                f.write(line)
        f.write(f"MYSQL_NODE={node}")


def ensure_mysql_running():
    """
    Run MySQL (or check it's running) and make sure .env is up-to-date.
    """
    job = get_or_start("mysql")
    update_dotenv_file(job.node)

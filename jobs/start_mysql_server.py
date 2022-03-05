import sys
import time
from pathlib import Path
from subprocess import run

import pandas as pd

bdqm_root = Path(__file__).parent.parent.resolve()


def get_or_start_mysql_job():
    qstat_out = run(
        "qstat -u $USER -n1", shell=True, capture_output=True
    ).stdout.decode("utf-8")

    jobs = pd.DataFrame(
        [r.split() for r in qstat_out.splitlines() if r and r[0].isnumeric()],
        columns=[
            "job_id",
            "username",
            "queue",
            "jobname",
            "sessid",
            "nds",
            "tsk",
            "memory",
            "time",
            "status",
            "elapsed",
            "node",
        ],
    )

    jobs["job_id"] = jobs.job_id.apply(lambda x: x.split(".")[0])

    mysql_jobs = jobs[(jobs.jobname == "mysqldb") & (jobs.status.isin(["Q", "R"]))]
    if len(mysql_jobs) == 1:
        mysql_job = mysql_jobs.iloc[0]
        if mysql_job.status == "Q":
            print(f"Waiting for MySQL job {mysql_job.job_id} to start...")
            time.sleep(10)
            return get_or_start_mysql_job()
        print(f"MySQL running, job ID: {mysql_job.job_id}")
        return mysql_job
    elif len(mysql_jobs) > 1:
        print(f"More than 1 mysql jobs running - kill them all!")
        sys.exit(1)
    else:
        print("Starting MySQL job")
        run(f"qsub {str(bdqm_root / 'jobs/mysql.pbs')}", shell=True)
        return get_or_start_mysql_job()


def update_dotenv_file(node):
    with open(bdqm_root / ".env") as f:
        lines = f.readlines()

    with open(bdqm_root / ".env", "w") as f:
        for line in lines:
            if not line.startswith("MYSQL_NODE"):
                f.write(line)
        f.write(f"MYSQL_NODE={node}")


if __name__ == "__main__":
    mysql_job = get_or_start_mysql_job()
    update_dotenv_file(mysql_job.node)

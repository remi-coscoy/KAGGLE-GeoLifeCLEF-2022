#!/usr/bin/python

import os
import sys
import subprocess
import tempfile


def makejob(n_samples, n_estimators, max_depth):
    return f"""#!/bin/bash

#SBATCH --job-name=rf
#SBATCH --nodes=1
#SBATCH --partition=gpu_prod_long
#SBATCH --time=24:00:00
#SBATCH --output=logslurms/slurm-%A_%a.out
#SBATCH --error=logslurms/slurm-%A_%a.err


current_dir=pwd
export PATH=$PATH:~/.local/bin

echo "Session " ${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}

echo "Running on " $(hostname)

echo "Copying the source directory and data"
date
mkdir $TMPDIR/code
rsync -r --exclude logs --exclude logslurms --exclude configs --exclude venv . $TMPDIR/code


echo "Setting up the virtual environment"
python3 -m venv venv
source venv/bin/activate

# Install the library
python -m pip install .


echo "Starting function"
python -m torchtmpl.random_forest {n_samples} {n_estimators} {max_depth}

if [[ $? != 0 ]]; then
    exit -1
fi
"""


def submit_job(job):
    with open("job.sbatch", "w") as fp:
        fp.write(job)
    os.system("sbatch job.sbatch")


# Ensure the log directory exists
os.system("mkdir -p logslurms")


if len(sys.argv) >= 2:
    n_samples = sys.argv[1]
    n_estimators = sys.argv[2]
    max_depth = sys.argv[3]
else:
    n_samples= 5000
    n_estimators=16
    max_depth= 10

# Launch the batch jobs
submit_job(makejob(n_samples, n_estimators, max_depth))
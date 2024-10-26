#!/usr/bin/python

import os
import sys
import subprocess
import tempfile


def makejob(commit_id,config,model_number):
    return f"""#!/bin/bash

#SBATCH --job-name=eval_model
#SBATCH --nodes=1
#SBATCH --partition=gpu_prod_long
#SBATCH --time=36:00:00
#SBATCH --output=logslurms/slurm-%A_%a.out
#SBATCH --error=logslurms/slurm-%A_%a.err
#SBATCH --array=1-1

current_dir=`pwd`
export PATH=$PATH:~/.local/bin

echo "Session " ${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}

echo "Running on " $(hostname)

echo "Copying the source directory and data"
date
mkdir $TMPDIR/code
rsync -r --exclude logs --exclude logslurms --exclude configs . $TMPDIR/code

echo "Checking out the correct version of the code commit_id {commit_id}"
cd $TMPDIR/code
git checkout {commit_id}


echo "Setting up the virtual environment"
python3 -m venv venv
source venv/bin/activate

# Install the library
python -m pip install .


echo "Training"
python -m torchtmpl.main {config} test {model_number}

if [[ $? != 0 ]]; then
    exit -1
fi
"""


def submit_job(job):
    with open("job.sbatch", "w") as fp:
        fp.write(job)
    os.system("sbatch job.sbatch")


# Ensure all the modified files have been staged and commited
# This is to guarantee that the commit id is a reliable certificate
# of the version of the code you want to evaluate
result = int(
    subprocess.run(
        "expr $(git diff --name-only | wc -l) + $(git diff --name-only --cached | wc -l)",
        shell=True,
        stdout=subprocess.PIPE,
    ).stdout.decode()
)
if result > 0:
    print(f"We found {result} modifications either not staged or not commited")
    raise RuntimeError(
        "You must stage and commit every modification before submission "
    )

commit_id = subprocess.check_output(
    "git log --pretty=format:'%H' -n 1", shell=True
).decode()

print(f"I will be using the commit id {commit_id}")

# Ensure the log directory exists
os.system("mkdir -p logslurms")

if len(sys.argv) != 2:
    print(f"Usage : {sys.argv[0]} model_name_and_number")
    sys.exit(-1)

model_number = sys.argv[1]

configpath = "logs/" + model_number + "/config.yaml"


# Copy the config in a temporary config file
os.system("mkdir -p configs")
tmp_configfilepath = tempfile.mkstemp(dir="./configs", suffix="-config.yml")[1]
os.system(f"cp {configpath} {tmp_configfilepath}")

# Launch the batch jobs
submit_job(makejob(commit_id, tmp_configfilepath,model_number))

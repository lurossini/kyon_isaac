
import argparse
import os
import sys
import wandb_inspector
import wandb


run_path : str = "arturo-laurenzi-istituto-italiano-di-tecnologia/isaaclab/8xjaku4i"


# Using the public API (works for runs you have access to)
api = wandb.Api()
run = api.run(run_path)

# Create directory and change to it
dir_name = run_path.replace("/", "_")
os.makedirs(dir_name, exist_ok=True)
os.chdir(dir_name)

insp = wandb_inspector.IsaacLabWandbInspector(run)
print("Remote:", insp.repo_remote)
print("SHA1  :", insp.sha1)
print("Diff  :\n", (insp.get_diff_text() or "")[:200], "...")
print("Checkpoints num:", len(insp.list_checkpoints()))
print("Last Checkpoint:", insp.get_last_checkpoint())

path = insp.download_checkpoint(insp.get_last_checkpoint().name, dest="checkpoint")
print("Downloaded last checkpoint to:", path)

# Save diff to a file
with open("diff.patch", "w") as f:
    f.write(insp.get_diff_text())
    f.write("\n")

# Script to run the downloaded checkpoint
play_inference_script = f"""
set -e
pip install -e /workspace/dir/pkg
cd /workspace/dir/pkg/scripts/
"""

with open("run_inference.sh", "w") as f:
    f.write(play_inference_script)


bash_script = f"""
#!/bin/bash
set -e 

# path to current script's directory
DIR="$( cd "$( dirname "${{BASH_SOURCE[0]}}" )" &> /dev/null && pwd )"
cd $DIR

# clone the repo at the specific commit (if not already present)
ls pkg || git clone {insp.repo_remote} pkg
cd pkg
git checkout {insp.sha1}

# apply diff (unless there are uncommitted changes)
git diff-index --quiet HEAD -- && git apply ../diff.patch

# env
export ACCEPT_EULA=Y

docker run --gpus all -it --rm \\
    -v .:/workspace/dir \\
    -e ACCEPT_EULA \\
    --entrypoint bash \\
    isaac-lab-base 
"""

with open("run_docker.sh", "w") as f:
    f.write(bash_script)

import subprocess
subprocess.run(["chmod", "+x", "run_docker.sh"])
print("Created run_docker.sh to run the checkpoint in a Docker container.")

subprocess.run(["bash", "run_docker.sh"])
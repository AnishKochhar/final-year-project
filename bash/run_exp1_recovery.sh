#!/bin/bash
#SBATCH --job-name=param_recovery_exp1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpgpuC
#SBATCH --time=4-00:00:00
#SBATCH --output=logs/exp1.out

#SBATCH --mail-type=ALL
#SBATCH --mail-user=ank121@ic.ac.uk

. /vol/cuda/12.4.0/setup.sh
source /vol/bitbucket/${USER}/dlenv/bin/activate
cd /vol/bitbucket/${USER}/whobpyt


python -u -m experiments.exp1.grid_search --rep 3

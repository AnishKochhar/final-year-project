#!/bin/bash
#SBATCH --job-name=rww_optuna
#SBATCH --gres=gpu:1          
#SBATCH --cpus-per-task=8     
#SBATCH --partition=gpgpuC
#SBATCH --time=4-00:00:00
#SBATCH --output=logs/exp5.out

#SBATCH --mail-type=ALL
#SBATCH --mail-user=ank121@ic.ac.uk

. /vol/cuda/12.4.0/setup.sh
source /vol/bitbucket/$USER/dlenv/bin/activate
cd /vol/bitbucket/$USER/whobpyt

python -u -m simulators.optuna_search_rww \
       --study-name whobpyt_pop_train \
       --trials 60 --n-jobs 1 \
       --data-root "/vol/bitbucket/$USER/fyp/HCP Data"

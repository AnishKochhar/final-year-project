#!/bin/bash
#SBATCH --job-name=rww_optuna
#SBATCH --gres=gpu:1          
#SBATCH --cpus-per-task=8     
#SBATCH --partition=gpgpuC
#SBATCH --time=4-00:00:00
#SBATCH --output=logs/exp3.out

#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=ank121@ic.ac.uk

. /vol/cuda/12.4.0/setup.sh
source /vol/bitbucket/$USER/dlenv/bin/activate
cd /vol/bitbucket/$USER/whobpyt

python -u -m experiments.regularizer_ablation --epochs 35 --lr 0.001 --g 85

#!/bin/bash
#SBATCH --job-name=exp3_reg
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

python -u -m experiments.regularizer_ablation --subject 42 --trials 50 --epochs 35 --study-name e3_optuna
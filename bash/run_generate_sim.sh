#!/bin/bash
#SBATCH --job-name=gen_sim
#SBATCH --gres=gpu:1          
#SBATCH --cpus-per-task=8     
#SBATCH --partition=gpgpuC
#SBATCH --time=4-00:00:00
#SBATCH --output=logs/generate.out

#SBATCH --mail-type=ALL
#SBATCH --mail-user=ank121@ic.ac.uk

. /vol/cuda/12.4.0/setup.sh
source /vol/bitbucket/$USER/dlenv/bin/activate
cd /vol/bitbucket/$USER/whobpyt

python -u -m simulators.generate_sim_fc_test --subj 18
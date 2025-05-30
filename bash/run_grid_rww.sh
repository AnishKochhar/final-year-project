#!/bin/bash
#SBATCH --job-name=grid_rww
#SBATCH --gres=gpu:1
#SBATCH --partition=gpgpuC
#SBATCH --time=4-00:00:00
#SBATCH --output=grid_rww_tmp.out

#SBATCH --mail-type=ALL
#SBATCH --mail-user=ank121@ic.ac.uk

. /vol/cuda/12.4.0/setup.sh
source /vol/bitbucket/${USER}/dlenv/bin/activate
cd /vol/bitbucket/${USER}/whobpyt


python -u -m simulators.grid_search_rww --subject 42 --g 86.8 --step 0.087 --chunk 100 \
	--lr 0.032 --lambda_rate 0.046 --lambda_spec 0.017 --lambda_disp  --rate_reg --spec_reg --disp_reg

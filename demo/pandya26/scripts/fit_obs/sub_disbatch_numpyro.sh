#!/bin/bash
#SBATCH -J nuts
#SBATCH --ntasks-per-node=1
#SBATCH -N 4
#SBATCH -t 120:00:00
#SBATCH --exclusive
#SBATCH --constraint="[genoa|icelake]"
##SBATCH --constraint=icelake
#SBATCH -p cca

module load disBatch/beta

disBatch /mnt/ceph/users/vpandya/sapphire/demo/pandya26/scripts/fit_obs/disbatch_tasks_numpyro

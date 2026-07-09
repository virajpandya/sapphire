#!/bin/bash
#SBATCH -J mock111
#SBATCH --ntasks-per-node=1
#SBATCH -N 10
#SBATCH -t 12:00:00
#SBATCH --exclusive
#SBATCH --constraint="[genoa|icelake|rome]"
## SBATCH --constraint=genoa
#SBATCH -p cca

module load disBatch/beta

disBatch /mnt/ceph/users/vpandya/sapphire/demo/pandya26/scripts/fit_mocks/disbatch_tasks_adam
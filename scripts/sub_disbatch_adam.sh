#!/bin/bash
#SBATCH -J disbatch
#SBATCH --ntasks-per-node=1
#SBATCH -N 10
#SBATCH -t 5:00:00
#SBATCH --exclusive
#SBATCH --constraint="[genoa|icelake|rome]"
## SBATCH --constraint=genoa
#SBATCH -p cca

module load disBatch/beta

disBatch /mnt/ceph/users/vpandya/sapphire/scripts/disbatch_tasks_adam
#!/bin/bash
#SBATCH -J adam
#SBATCH --ntasks-per-node=1
#SBATCH -N 1
#SBATCH -t 10:00:00
#SBATCH --exclusive
#SBATCH --constraint="[genoa|icelake|rome]"
## SBATCH --constraint=genoa
#SBATCH -p gen

source /mnt/home/vpandya/.bashrc
source activate japphire

python -m sapphire --path_config "/mnt/ceph/users/vpandya/sapphire/demo/pandya26/scripts/fit_obs/config_obs_adam.yaml" &> adam_obs_111.log
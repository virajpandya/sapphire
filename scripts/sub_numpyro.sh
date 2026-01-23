#!/bin/bash
#SBATCH -J numpyro
#SBATCH --ntasks-per-node=1
#SBATCH -N 1
#SBATCH -t 10:00:00
#SBATCH --exclusive
#SBATCH --constraint="[genoa|icelake|rome]"
## SBATCH --constraint=genoa
#SBATCH -p cca
##SBATCH -o log_numpyro_obs.o%j

source /mnt/home/vpandya/.bashrc
source activate jaxclone

python -m sapphire --path_config "/mnt/ceph/users/vpandya/sapphire/scripts/config.yaml" --chain_num 0 --flag_smhm 1 --flag_fgas 1 --flag_mzr 1 #&> /mnt/ceph/users/vpandya/sapphire_outputs/logs/numpyro_chain${mocknum}_100.log
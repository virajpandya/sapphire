#!/bin/bash
#SBATCH -J adam
#SBATCH --ntasks-per-node=1
#SBATCH -N 1
#SBATCH -t 10:00:00
#SBATCH --exclusive
#SBATCH --constraint="[genoa|icelake|rome]"
## SBATCH --constraint=genoa
#SBATCH -p cca
##SBATCH -o log_numpyro_obs.o%j

source /mnt/home/vpandya/.bashrc
source activate japphire

python -m sapphire --path_config "/mnt/ceph/users/vpandya/sapphire/scripts/config.yaml" --flag_smhm 1 --flag_fgas 1 --flag_mzr 0 --flag_sfms 0 --flag_mzr_gas 1 --scale_err_smhm 1.0 --scale_err_fgas 1.0 --scale_err_mzr_gas 1.0 --output_suffix "{flag_smhm}{flag_fgas}{flag_mzr_gas}" &> /mnt/ceph/users/vpandya/sapphire_outputs/logs/adam_mix_111.log
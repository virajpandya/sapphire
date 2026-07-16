#!/bin/bash
#SBATCH -J aies100
#SBATCH --ntasks-per-node=1
#SBATCH -N 1
#SBATCH -t 96:00:00
#SBATCH --gpus=1
##SBATCH -C h100
#SBATCH --mem=300G
#SBATCH --cpus-per-task=32
#SBATCH -p gpu

source /mnt/home/vpandya/.bashrc
source activate japphiregpu

python -m sapphire --path_config "/mnt/ceph/users/vpandya/sapphire/demo/pandya26/scripts/fit_obs/config_obs_aies.yaml" &> aies20split_mix_100.log

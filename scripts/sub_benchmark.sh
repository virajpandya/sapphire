#!/bin/bash
#SBATCH -J benchgp8
#SBATCH --ntasks-per-node=1
#SBATCH -N 1
##SBATCH --exclusive
##SBATCH --constraint=icelake
#SBATCH --gpus=8
#SBATCH -C h100
#SBATCH --mem=1000G
#SBATCH --cpus-per-task=64
#SBATCH -p gpu
#SBATCH -t 96:00:00

source /mnt/home/vpandya/.bashrc
source activate japphiregpu

python -m sapphire --path_config "/mnt/ceph/users/vpandya/sapphire/scripts/config.yaml" --output_suffix "8gpu64cpu" &> /mnt/ceph/users/vpandya/sapphire_outputs/logs/benchmark_8gpu64cpu.log
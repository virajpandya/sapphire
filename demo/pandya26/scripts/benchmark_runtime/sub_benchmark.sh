#!/bin/bash
#SBATCH -J bench
#SBATCH --ntasks-per-node=1
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --constraint="[genoa|icelake|rome]"
##SBATCH --gpus=8
##SBATCH -C h100
##SBATCH --mem=1000G
##SBATCH --cpus-per-task=64
##SBATCH -p gpu
#SBATCH -p gen
#SBATCH -t 96:00:00

source /mnt/home/vpandya/.bashrc
# source activate japphiregpu
source activate japphire

python -m sapphire --path_config "/mnt/ceph/users/vpandya/sapphire/demo/pandya26/scripts/benchmark_runtime/config_benchmark.yaml" &> benchmark_64cpu.log
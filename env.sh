module load modules/2.2-20230808
module load python/3.10.10
module load slurm
#module load gcc/12.2.0 #gcc/10.3.0
# module load gcc/10.3.0
# module load gsl/2.7
# module load openmpi/4.0.7
# module load hdf5/mpi-1.8.22
# module load fftw/mpi-3.3.10
# #module load hwloc/2.9.0 #hwloc/2.7.1
# module load hwloc/2.7.1
# module load gmp/6.2.1

# for BOLA to work, need python <3.8. default on rusty is 3.6, so use that.
#module load python/3.9.12 #python/3.9.12

# this line is to use this package globally
# export PYTHONPATH="${PYTHONPATH}/mnt/home/bterrazas/ceph/arepo/pylib"
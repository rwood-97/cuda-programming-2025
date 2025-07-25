#!/bin/bash
# set the number of nodes and processes per node
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=short

# set max wallclock time
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1

# set name of job
#SBATCH --job-name=prac2

#SBATCH -o log_tensorCUBLAS_%j.out

# use our reservation
#SBATCH --reservation=cuda2025

module purge
module load CUDA

make clean
make

# run the executable
./tensorCUBLAS 32
./tensorCUBLAS 64
./tensorCUBLAS 128
./tensorCUBLAS 256
./tensorCUBLAS 512
./tensorCUBLAS 1024



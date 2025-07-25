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

# use our reservation
#SBATCH --reservation=cuda2025

# request extra memory
#SBATCH --mem 16G

module purge
module load CUDA

make clean
make

ncu --metrics "smsp__sass_thread_inst_executed_op_fp32_pred_on.sum, smsp__sass_thread_inst_executed_op_integer_pred_on.sum" laplace3d
# ./laplace3d 32 4


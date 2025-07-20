#!/bin/bash
# vim: et:ts=4:sts=4:sw=4

# Execute using:
# sbatch ./bask-local.sh

echo "## CUDA course initialisation script"

if [[ $(hostname) == *login* ]]; then
    echo "This script should only be run on a compute node"
    exit 0
fi

# Quit on error
#set -e

echo "## Loading modules"

module -q purge
module -q load baskerville
module -q load bask-apps/live
module -q load CUDAcore/11.1.1
module -q load CUDA/11.1.1-GCC-10.2.0

echo "## CUDA course initialisation script completed"

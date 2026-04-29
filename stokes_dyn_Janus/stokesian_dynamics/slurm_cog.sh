#!/bin/bash

#SBATCH -N 1
#SBATCH -c 1
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH -t 02-00

#SBATCH -p ug-gpu-small
#SBATCH --qos=short
#SBATCH --job-name=cogExperiment
#SBATCH --mail-type=ALL
#SBATCH --mail-user dbxl46@durham.ac.uk

#SBATCH -e stderr-file4
#SBATCH -o stdout-file4

source /etc/profile
module load intel-oneapi/2022.1.2/vtune
module load intel-oneapi/2022.1.2/mpi
module load intel-oneapi/2022.1.2/compiler

VENV=/home3/dbxl46/pytential_stokes/pytential_stokes/myenv
source $VENV/bin/activate

export PATH="$VENV/bin:$PATH"

python -c "import pyopencl as cl; print(cl.get_platforms())"
export PYOPENCL_CTX='0'
export PYOPENCL_COMPILER_OUTPUT='1'

# Run your script
mkdir cogExperiment
python3.11 -u -O run_simulation.py 14 11 0.1 100 fte 2 3 cogExperiment 30
python3.11 plotting/plot_positions.py 1 0 cogExperiment cogExperiment
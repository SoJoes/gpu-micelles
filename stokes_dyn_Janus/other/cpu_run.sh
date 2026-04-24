#!/bin/bash

#SBATCH -N 1
#SBATCH -c 1
#SBATCH -t 00-30

#SBATCH -p cpu
#SBATCH --qos=short
#SBATCH --job-name=relaxation

#SBATCH -e stderr-file
#SBATCH -o stdout-file

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
python3.11 -u singleParticlePlots.py
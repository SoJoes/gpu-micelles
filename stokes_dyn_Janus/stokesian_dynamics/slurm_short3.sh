#!/bin/bash

#SBATCH -N 1
#SBATCH -c 1
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH -t 01-00

#SBATCH -p ug-gpu-small
#SBATCH --qos=short
#SBATCH --job-name=grid

#SBATCH -e stderr-file3
#SBATCH -o stdout-file3

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

rm -rf frame_output3
mkdir -p frame_output3
python3.11 -u -O run_simulation.py 12 11 0.1 50 fte 2 1 output3 25
python3.11 -u -O run_simulation.py 12 11 0.1 50 fte 2 1 output3 25
python3.11 plotting/plot_positions.py 2 25 frame_output3 output3
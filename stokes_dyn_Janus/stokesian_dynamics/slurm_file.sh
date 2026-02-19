#!/bin/bash

#SBATCH -N 1
#SBATCH -c 1
#SBATCH --gres=gpu:1g.10gb:1

#SBATCH -p ug-gpu-small
#SBATCH --qos=debug
#SBATCH --job-name=micelle_formation
#SBATCH --mail-type=ALL
#SBATCH --mail-user dbxl46@durham.ac.uk

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

# Run your script

rm -rf frame_output
mkdir -p frame_output
python3.11 -u run_simulation.py 11 10 0.1 50 fte
python3.11 plotting/plot_positions.py
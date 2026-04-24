#!/bin/bash

#SBATCH -N 1
#SBATCH -c 1
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH -t 02-00

#SBATCH -p ug-gpu-small
#SBATCH --qos=short
#SBATCH --job-name=janusGrid
#SBATCH --mail-type=ALL
#SBATCH --mail-user dbxl46@durham.ac.uk

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
mkdir janusShear/009
python3.11 -u -O run_simulation.py 12 14 0.1 50 fte 5 1 janusRelax 30
python3.11 -u -O run_simulation.py 12 14 0.1 50 fte 5 1 janusRelax 30
echo "BEGINNNING TO SHEAR"
python3.11 -u -O run_simulation.py 20 14 0.1 50 fte 5 1 janusShear/009 30
for i in {0..3}; do
  echo "Started simulation run $i"
  python3.11 -u -O run_simulation.py 12 14 0.1 50 fte 5 1 janusShear/009 30
done
python3.11 plotting/plot_positions.py 5 0 janusShear/009 janusShear/009
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

#SBATCH -e janus2-err
#SBATCH -o janus2-out

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
echo "BEGINNNING TO SHEAR"
mkdir janusShear/002
python3.11 -u -O run_simulation.py 20 15 0.1 50 fte 5 1 janusShear/002 30
for i in {0..3}; do
  echo "Started simulation run $i"
  python3.11 -u -O run_simulation.py 12 15 0.1 50 fte 5 1 janusShear/002 30
done
python3.11 plotting/plot_positions.py 5 0 janusShear/002 janusShear/002
#!/bin/bash

#SBATCH -N 1
#SBATCH -c 1
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH -t 02-00

#SBATCH -p ug-gpu-small
#SBATCH --qos=short
#SBATCH --job-name=cogGrid2
#SBATCH --mail-type=ALL
#SBATCH --mail-user dbxl46@durham.ac.uk

#SBATCH -e cog2-err
#SBATCH -o cog2-out

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
mkdir cogShear/0059
python3.11 -u -O run_simulation.py 22 20 0.1 50 fte 5 3 cogShear/0059 30
for i in {0..3}; do
  echo "Started simulation run $i"
  python3.11 -u -O run_simulation.py 12 20 0.1 50 fte 5 3 cogShear/0059 30
done
python3.11 plotting/plot_positions.py 5 0 cogShear/0059 cogShear/0059
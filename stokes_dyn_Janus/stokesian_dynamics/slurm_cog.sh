#!/bin/bash

#SBATCH -N 1
#SBATCH -c 1
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH -t 02-00

#SBATCH -p ug-gpu-small
#SBATCH --qos=short
#SBATCH --job-name=cogGrid

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

rm -rf frame_output4
mkdir -p frame_output4
mkdir -p output4
python3.11 -u -O run_simulation.py 17 11 0.1 50 fte 5 3 output4 25
for i in {0..18}; do
  echo "Started simulation run $i"
  python3.11 -u -O run_simulation.py 12 11 0.1 50 fte 5 3 output4 25
done
for i in {0...19}; do
  echo "Started simulation run $i"
  python3.11 -u -O run_simulation.py 12 13 0.1 50 fte 5 3 output4 25
done
python3.11 plotting/plot_positions.py 40 0 frame_output4 output4
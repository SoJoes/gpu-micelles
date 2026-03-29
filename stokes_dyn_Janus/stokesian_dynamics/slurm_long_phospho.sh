#!/bin/bash

#SBATCH -N 1
#SBATCH -c 1
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH -t 01-00

#SBATCH -p ug-gpu-small
#SBATCH --qos=short
#SBATCH --job-name=phospho
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

rm -rf frame_output1
mkdir -p frame_output1
mkdir -p output1
python3.11 -O run_simulation.py 16 10 0.05 80 fte 10 1 output1
for i in {0..5}; do
    echo "Started simulation run $i"
    python3.11 -O run_simulation.py 12 10 0.05 80 fte 10 1 output1
done
python3.11 plotting/plot_positions.py 6 0 frame_output2
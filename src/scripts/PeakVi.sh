#!/bin/bash
#BSUB -J Peak_Script_2
#BSUB -o outfiles/Peak_Script_2%J.out
#BSUB -e outfiles/Peak_Script_2%J.err
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "rusage[mem=60GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 48:00
# end of BSUB options

cd "$LS_SUBCWD" || exit 1

module purge
module load python3/3.12.11

source .venv-cpu/bin/activate

export OMP_NUM_THREADS=$LSB_DJOB_NUMPROC
export OPENBLAS_NUM_THREADS=$LSB_DJOB_NUMPROC
export MKL_NUM_THREADS=$LSB_DJOB_NUMPROC
export NUMEXPR_NUM_THREADS=$LSB_DJOB_NUMPROC
export MKL_DYNAMIC=false
export OMP_DYNAMIC=false


python src/PeakVI/main_peakvi.py

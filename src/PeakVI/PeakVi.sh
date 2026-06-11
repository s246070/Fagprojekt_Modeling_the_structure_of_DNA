#!/bin/bash
#BSUB -J Peak_Script_2
#BSUB -o outfiles/Peak_Script_2%J.out
#BSUB -e outfiles/Peak_Script_2%J.err
#BSUB -q gpua100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 24:00
# end of BSUB options

cd "$LS_SUBCWD" || exit 1

module purge
module load python3/3.12.11
module load cuda/12.8.1

source .venv/bin/activate

export OMP_NUM_THREADS=$LSB_DJOB_NUMPROC
export OPENBLAS_NUM_THREADS=$LSB_DJOB_NUMPROC
export MKL_NUM_THREADS=$LSB_DJOB_NUMPROC
export NUMEXPR_NUM_THREADS=$LSB_DJOB_NUMPROC
export MKL_DYNAMIC=false
export OMP_DYNAMIC=false


python src/PeakVI/main_peakvi.py

#!/bin/bash
#BSUB -J GPU_PeakVi
#BSUB -o outfiles/GPU_PeakVi_%J.out
#BSUB -e outfiles/GPU_PeakVi_%J.err
#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -n 4
#BSUB -R "rusage[mem=2GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 24:00
# end of BSUB options

cd "$LS_SUBCWD" || exit 1

module purge
module load python3/3.12.11
module load cuda/12.6.1

source .venv-gpu/bin/activate

export OMP_NUM_THREADS=$LSB_DJOB_NUMPROC
export OPENBLAS_NUM_THREADS=$LSB_DJOB_NUMPROC
export MKL_NUM_THREADS=$LSB_DJOB_NUMPROC
export NUMEXPR_NUM_THREADS=$LSB_DJOB_NUMPROC
export MKL_DYNAMIC=false
export OMP_DYNAMIC=false


python src/PeakVI/main_peakvi.py

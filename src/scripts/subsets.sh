#!/bin/bash
#BSUB -J subsets
#BSUB -o outfiles/subsets%J.out
#BSUB -e outfiles/subsets%J.err
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "rusage[mem=100GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 24:00
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

python src/dna_modelling/subsets.py 
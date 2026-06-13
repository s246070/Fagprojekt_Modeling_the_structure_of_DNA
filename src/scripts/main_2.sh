#!/bin/bash
#BSUB -J batch_Script_2
#BSUB -o outfiles/batch_Script_2%J.out
#BSUB -e outfiles/batch_Script_2%J.err
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "rusage[mem=50GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 26:00
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

python src/dna_modelling/main_2.py \
	--seed 1 \
	--ls-dim 16 \
	--index 1616 \
	--no-weighting \
	--epochs 1001 \
	--lr 0.005 \
	--batching \
	--validation \
	--num-blocks 100
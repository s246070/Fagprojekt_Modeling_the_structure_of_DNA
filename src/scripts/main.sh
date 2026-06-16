#!/bin/bash
#BSUB -J TEST_Script_2
#BSUB -o outfiles/TEST_Script_2%J.out
#BSUB -e outfiles/TEST_Script_2%J.err
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "rusage[mem=175GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 17:00
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

python src/dna_modelling/main.py \
	--seed 1 \
	--ls-dim 3 \
	--index 3333 \
	--full-data \
	--epochs 1001 \
	--lr 0.03 \
	--no-weighting \
	--no-batching \
	--validation \
	--num-blocks 1000
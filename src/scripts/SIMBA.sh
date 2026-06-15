#!/bin/bash
#BSUB -J SIMBA_full_dim50
#BSUB -o outfiles/SIMBA_full_dim50%J.out
#BSUB -e outfiles/SIMBA_full_dim50%J.err
#BSUB -q hpc
#BSUB -n 8
#BSUB -R "rusage[mem=80GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 48:00
# end of BSUB options

cd "$LS_SUBCWD" || exit 1

module purge
module load python3/3.10.20

source .venv-simba/bin/activate

export OMP_NUM_THREADS=$LSB_DJOB_NUMPROC
export OPENBLAS_NUM_THREADS=$LSB_DJOB_NUMPROC
export MKL_NUM_THREADS=$LSB_DJOB_NUMPROC
export NUMEXPR_NUM_THREADS=$LSB_DJOB_NUMPROC
export MKL_DYNAMIC=false
export OMP_DYNAMIC=false

python src/simba/main_simba.py \
	--full \
	--run-name simba_full_dim50 \
	--seed 1 \
	--embedding-dim 50 \
	--workers "$LSB_DJOB_NUMPROC" \
	--num-epochs 10 \
	--min-cells 3
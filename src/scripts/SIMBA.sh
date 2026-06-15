#!/bin/bash
#BSUB -J SIMBA_subset_dim2
#BSUB -o outfiles/SIMBA_subset_dim2%J.out
#BSUB -e outfiles/SIMBA_subset_dim2%J.err
#BSUB -q hpc
#BSUB -n 8
#BSUB -R "rusage[mem=10GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 48:00
# end of BSUB options

cd "$LS_SUBCWD" || exit 1

module purge

source .venv-simba/bin/activate

# Put SIMBA/PBG temporary files on /work3 instead of home/project quota.
export SIMBA_WORKDIR="/work3/atde/$USER/simba_${LSB_JOBID}"
mkdir -p "$SIMBA_WORKDIR"

df -h "$SIMBA_WORKDIR"

export OMP_NUM_THREADS=$LSB_DJOB_NUMPROC
export OPENBLAS_NUM_THREADS=$LSB_DJOB_NUMPROC
export MKL_NUM_THREADS=$LSB_DJOB_NUMPROC
export NUMEXPR_NUM_THREADS=$LSB_DJOB_NUMPROC
export MKL_DYNAMIC=false
export OMP_DYNAMIC=false

python src/simba/main_simba.py \
	--run-name simba_subset_dim2 \
	--seed 1 \
	--embedding-dim 2 \
	--workers "$LSB_DJOB_NUMPROC" \
	--num-epochs 10 \
	--min-cells 3 \
	--specify-path train_sets/adata_subset_10k_1.h5ad \
	--workdir "$SIMBA_WORKDIR"
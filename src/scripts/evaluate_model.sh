#!/bin/bash
#BSUB -J Evaluate_Script_2
#BSUB -o outfiles/Evaluate_Script_2%J.out
#BSUB -e outfiles/Evaluate_Script_2%J.err
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "rusage[mem=150GB]"
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

models_path="models/plotting/ldm_ls2_weighting_False_run4000.pth"
DIM=2
SEED=1
INDEX=4000
EPOCHS=1001
LR=0.0025
NUM_BLOCKS=10000

python src/dna_modelling/evaluate_models.py \
  --seed $SEED \
  --index $INDEX \
  --full-data \
  --epochs $EPOCHS \
  --lr $LR \
  --no-weighting \
  --no-batching \
  --validation \
  --num-blocks $NUM_BLOCKS \
  --ls_dim $DIM \
  --model-path $models_path
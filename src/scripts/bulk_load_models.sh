#!/bin/bash
# Submit multiple LSF jobs with different LS_DIM and WEIGHTING parameters

set -e

# Define the parameter ranges
models_paths=("models/ldm_ls2_epoch1000_blocks10000_index1.pth" "models/ldm_ls3_epoch1000_blocks10000_index1.pth" "models/ldm_ls8_epoch1000_blocks10000_index1.pth" "models/ldm_ls16_epoch1000_blocks10000_index1.pth" "models/ldm_ls32_epoch1000_blocks10000_index1.pth")
DIM=(2 3 8 16 32)
SEED=1
INDEX=2
EPOCHS=1001
LR=0.0025
NUM_BLOCKS=10000
DATA_PATH="train_sets/adata_subset_10k_1.h5ad"

echo "Submitting LSF jobs..."
echo "========================================"

for i in "${!models_paths[@]}"; do
    model_path="${models_paths[$i]}"
    job_name="JOHN"
    echo "Submitting: $job_name"

  job_cmd=$(cat <<EOF
cd "$PWD" || exit 1
module purge
module load python3/3.12.11
source .venv-cpu/bin/activate
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export MKL_DYNAMIC=false
export OMP_DYNAMIC=false
python src/dna_modelling/load_models.py \
  --seed $SEED \
  --index $INDEX \
  --no-full-data \
  --epochs $EPOCHS \
  --lr $LR \
  --no-weighting \
  --no-batching \
  --validation \
  --num-blocks $NUM_BLOCKS \
  --data-path $DATA_PATH \
  --ls_dim ${DIM[$i]} \
  --model-path $model_path
EOF
  )

  bsub \
    -J "$job_name" \
    -o "outfiles/${job_name}%J.out" \
    -e "outfiles/${job_name}%J.err" \
    -q hpc \
    -n 4 \
    -R "rusage[mem=50GB]" \
    -R "span[hosts=1]" \
    -W 48:00 \
    bash -lc "$job_cmd"
done

echo "========================================"
echo "All jobs submitted! Check status with: bstat"

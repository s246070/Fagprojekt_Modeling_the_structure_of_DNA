#!/bin/bash
# Submit 10 LSF jobs for PeakVI, each with a different random seed and 6000 epochs.

set -e

SEEDS=(1 2 3 4 5 6 7 8 9 10)
EPOCHS=6000
INTERVAL=50
QUEUE=hpc
NUM_PROC=4
MEM=9900
WALLTIME=16:00

echo "Submitting PeakVI jobs..."
echo "========================================"

for seed in "${SEEDS[@]}"; do
  job_name="PeakVI_seed_${seed}"
  echo "Submitting: $job_name (SEED=$seed, EPOCHS=$EPOCHS)"

  job_cmd=$(cat <<EOF
cd "$PWD" || exit 1
module purge
module load python3/3.12.11
source .venv-cpu/bin/activate
export OMP_NUM_THREADS=$NUM_PROC
export OPENBLAS_NUM_THREADS=$NUM_PROC
export MKL_NUM_THREADS=$NUM_PROC
export NUMEXPR_NUM_THREADS=$NUM_PROC
export MKL_DYNAMIC=false
export OMP_DYNAMIC=false
export INDEX=$seed
export SEED=$seed
export EPOCHS=$EPOCHS
export INTERVAL=$INTERVAL
export MODEL_DIR="models/peakvi_seed${seed}"
python src/PeakVI/main_peakvi.py
EOF
  )

  bsub \
    -J "$job_name" \
    -o "outfiles/${job_name}%J.out" \
    -e "outfiles/${job_name}%J.err" \
    -q "$QUEUE" \
    -n "$NUM_PROC" \
    -R "rusage[mem=${MEM}]" \
    -R "span[hosts=1]" \
    -W "$WALLTIME" \
    bash -lc "$job_cmd"

done

echo "========================================"
echo "All PeakVI jobs submitted! Check status with: bstat"

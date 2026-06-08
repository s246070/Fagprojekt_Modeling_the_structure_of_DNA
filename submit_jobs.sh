#!/bin/bash
# Submit multiple LSF jobs with different LS_DIM and WEIGHTING parameters

set -e

# Define the parameter ranges
LS_DIMS=(2 3 8 16 32)
WEIGHTINGS=(true false)

echo "Submitting LSF jobs..."
echo "========================================"

for dim in "${LS_DIMS[@]}"; do
  for weighting in "${WEIGHTINGS[@]}"; do
    for idx in {1..10}; do
      job_name="Main_Script_${dim}_dim_weighting_${weighting}_run${idx}"
      echo "Submitting: $job_name (LS_DIM=$dim, WEIGHTING=$weighting, INDEX=$idx)"
    
    bsub \
      -J "$job_name" \
      -o "outfiles/${job_name}%J.out" \
      -e "outfiles/${job_name}%J.err" \
      -q hpc \
      -n 4 \
      -R "rusage[mem=9900]" \
      -R "span[hosts=1]" \
      -W 08:00 \
      bash -c "
        cd '$PWD' || exit 1
        module purge
        module load python3/3.12.11
        source .venv/bin/activate
        export OMP_NUM_THREADS=4
        export OPENBLAS_NUM_THREADS=4
        export MKL_NUM_THREADS=4
        export NUMEXPR_NUM_THREADS=4
        export MKL_DYNAMIC=false
        export OMP_DYNAMIC=false
        export LS_DIM=$dim
        export INDEX=$idx
        export WEIGHTING=$weighting
        export SEED=$idx
        python src/dna_modelling/main.py
      "
    done
  done
done

echo "========================================"
echo "All jobs submitted! Check status with: bstat"

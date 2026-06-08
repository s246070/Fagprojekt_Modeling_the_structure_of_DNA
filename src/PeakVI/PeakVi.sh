#!/bin/bash
#BSUB -J PeakVI_Main
#BSUB -o outfiles/PeakVI_Main%J.out
#BSUB -e outfiles/PeakVI_Main%J.err
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "rusage[mem=9900]"
#BSUB -R "span[hosts=1]"
#BSUB -W 08:00
# end of BSUB options

cd "$LS_SUBCWD" || exit 1

module purge
module load python3/3.12.11

# Activate project virtualenv (relative to job cwd)
if [ -f "$LS_SUBCWD/.venv/bin/activate" ]; then
	source "$LS_SUBCWD/.venv/bin/activate"
elif [ -f ".venv/bin/activate" ]; then
	source .venv/bin/activate
fi

export OMP_NUM_THREADS=${LSB_DJOB_NUMPROC:-4}
export OPENBLAS_NUM_THREADS=${LSB_DJOB_NUMPROC:-4}
export MKL_NUM_THREADS=${LSB_DJOB_NUMPROC:-4}
export NUMEXPR_NUM_THREADS=${LSB_DJOB_NUMPROC:-4}
export MKL_DYNAMIC=false
export OMP_DYNAMIC=false

# Job-specific environment (can be overridden by bsub -env or export before submission)
export INDEX=${INDEX:-1}
export EPOCHS=${EPOCHS:-200}
export INTERVAL=${INTERVAL:-50}
export MODEL_DIR=${MODEL_DIR:-models/peakvi_run${INDEX}}

# Ensure python finds project modules
export PYTHONPATH="$LS_SUBCWD:$PYTHONPATH"
export PYTHONUNBUFFERED=1

mkdir -p outfiles results models

python -u src/PeakVI/main_peakvi.py

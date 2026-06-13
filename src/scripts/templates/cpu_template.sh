#!/bin/bash
#BSUB -J CPU_Job
#BSUB -o outfiles/CPU_Job_%J.out
#BSUB -e outfiles/CPU_Job_%J.err
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "rusage[mem=30GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 24:00
# end of BSUB options

cd "$LS_SUBCWD" || exit 1

mkdir -p outfiles

module purge
module load python3/3.12.11

source .venv-cpu/bin/activate

export OMP_NUM_THREADS=$LSB_DJOB_NUMPROC
export OPENBLAS_NUM_THREADS=$LSB_DJOB_NUMPROC
export MKL_NUM_THREADS=$LSB_DJOB_NUMPROC
export NUMEXPR_NUM_THREADS=$LSB_DJOB_NUMPROC
export MKL_DYNAMIC=false
export OMP_DYNAMIC=false

echo "Running CPU job"
echo "Host: $(hostname)"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
echo "Threads: $LSB_DJOB_NUMPROC"

python -c "import torch; print('Torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

python src/path_to_your_script.py

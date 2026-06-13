#!/bin/bash
#BSUB -J GPU_Job
#BSUB -o outfiles/GPU_Job_%J.out
#BSUB -e outfiles/GPU_Job_%J.err
#BSUB -q gpua100
#BSUB -gpu "num=1"
#BSUB -n 4
#BSUB -R "rusage[mem=60GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 48:00
# end of BSUB options

cd "$LS_SUBCWD" || exit 1

mkdir -p outfiles

module purge
module load python3/3.12.11
module load cuda/12.8.1

source .venv-gpu/bin/activate

export OMP_NUM_THREADS=$LSB_DJOB_NUMPROC
export OPENBLAS_NUM_THREADS=$LSB_DJOB_NUMPROC
export MKL_NUM_THREADS=$LSB_DJOB_NUMPROC
export NUMEXPR_NUM_THREADS=$LSB_DJOB_NUMPROC
export MKL_DYNAMIC=false
export OMP_DYNAMIC=false

echo "Running GPU job"
echo "Host: $(hostname)"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
echo "Threads: $LSB_DJOB_NUMPROC"

nvidia-smi

python -c "import torch; print('Torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA device count:', torch.cuda.device_count()); print('CUDA device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

python src/path_to_your_script.py

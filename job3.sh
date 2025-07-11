#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-15
#SBATCH --job-name=cqa-gemma_quantisation_6k
#SBATCH --mem=10GB

module purge

module load Python/3.10.4-GCCcore-11.3.0


# Create venv if not exists
if [ ! -d "thesis_venv" ]; then
  python3 -m venv thesis_venv
fi

source ./thesis_venv/bin/activate

# having automatic git pull when submitting multiple jobs creates problems when the jobs are queued -
# e.g. all jobs are run with the same model if model argument got changed after the job submission
#git pull origin main

pip install --upgrade pip
pip install --quiet -r requirements.txt

ROWS_PER_TASK=100

# Compute index range for this SLURM array task
START_INDEX=$((SLURM_ARRAY_TASK_ID * ROWS_PER_TASK))
END_INDEX=$(((SLURM_ARRAY_TASK_ID + 1) * ROWS_PER_TASK))

# Prepare env
export HF_HOME=/tmp

# Run the script with args
python -m src.main --model gemma9b --backend llama_cpp --start ${START_INDEX} --end ${END_INDEX} --task CommonsenseQA --quantisation 6

deactivate

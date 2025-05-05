#!/bin/bash
#SBATCH --time=06:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-9
#SBATCH --job-name=svamp_perturb_gemma2
#SBATCH --mem=50GB

module purge

module load Python/3.10.4-GCCcore-11.3.0


# Create venv if not exists
if [ ! -d "thesis_venv" ]; then
  python3 -m venv thesis_venv
fi

source ./thesis_venv/bin/activate

git pull origin main

pip install --upgrade pip
pip install -r requirements.txt

ROWS_PER_TASK=100

# Compute index range for this SLURM array task
START_INDEX=$((SLURM_ARRAY_TASK_ID * ROWS_PER_TASK))
END_INDEX=$(((SLURM_ARRAY_TASK_ID + 1) * ROWS_PER_TASK))

# Run the script with args
python -m src.main --model gemma27b --start ${START_INDEX} --end ${END_INDEX} --quantization True

deactivate
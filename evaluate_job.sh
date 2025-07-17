#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks=1
#SBATCH --array=0,2,4,6,8
#SBATCH --job-name=eval-svamp-deepseek-quant4-part1
#SBATCH --mem=10GB

module purge

module load Python/3.10.4-GCCcore-11.3.0


# Create venv if not exists
if [ ! -d "thesis_venv" ]; then
  python3 -m venv thesis_venv
fi

source ./thesis_venv/bin/activate


pip install --upgrade pip
pip install --quiet -r requirements.txt

ROWS_PER_TASK=100

# Compute index range for this SLURM array task
START_INDEX=$((SLURM_ARRAY_TASK_ID * ROWS_PER_TASK))

export HF_HOME=/tmp
# Run the script with args
python -m src.evaluate_uncertainty --model deepseek --index ${START_INDEX} --method cosine --task SVAMP --quantisation 4

deactivate

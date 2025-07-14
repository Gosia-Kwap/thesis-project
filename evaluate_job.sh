#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks=1
#SBATCH --array=1-2
#SBATCH --job-name=test_eval_ai2arc_llama_quant4
#SBATCH --mem=10GB

set -euo pipefail
PS4='[\D{%T}] ' ; set -x

module purge

module load Python/3.10.4-GCCcore-11.3.0


# Create venv if not exists
if [ ! -d "thesis_venv" ]; then
  python3 -m venv thesis_venv
fi

source ./thesis_venv/bin/activate


pip install --upgrade pip
pip install -r requirements.txt

ROWS_PER_TASK=100

# Compute index range for this SLURM array task
START_INDEX=$((SLURM_ARRAY_TASK_ID * ROWS_PER_TASK))

export HF_HOME=/tmp
# Run the script with args
python -X importtime -m src.evaluate_uncertainty --model llama3 --index ${START_INDEX} --method cosine --task ai2_arc --quantisation 4

deactivate

#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks=1
#SBATCH --job-name=evaluate_results_llama
#SBATCH --mem=10GB
ś
module purge

module load Python/3.10.4-GCCcore-11.3.0


# Create venv if not exists
if [ ! -d "thesis_venv" ]; then
  python3 -m venv thesis_venv
fi

source ./thesis_venv/bin/activate


pip install --upgrade pip
pip install --quiet -r requirements.txt

export HF_HOME=/tmp


# Run the script with args
python -m src.rephrase_questions --task CommonsenseQA

deactivate
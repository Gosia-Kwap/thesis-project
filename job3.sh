#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-15
#SBATCH --job-name=cqa-deepseek
#SBATCH --mem=10GB

module purge

module load Python/3.10.4-GCCcore-11.3.0


# Create venv if not exists
if [ ! -d "thesis_venv_deepseek" ]; then
  python3 -m venv thesis_venv_deepseek
fi

source ./thesis_venv_deepseek/bin/activate

pip install --upgrade pip
pip install --quiet -r requirements_deepseek.txt

ROWS_PER_TASK=100

# Compute index range for this SLURM array task
START_INDEX=$((SLURM_ARRAY_TASK_ID * ROWS_PER_TASK))
END_INDEX=$(((SLURM_ARRAY_TASK_ID + 1) * ROWS_PER_TASK))

# Prepare env
export HF_HOME=/tmp

# Run the script with args
#python -m src.main --model deepseek --backend llama_cpp --start ${START_INDEX} --end ${END_INDEX} --task logiqa --quantisation 4
python -m src.main --model deepseek --start ${START_INDEX} --end ${END_INDEX} --task CommonsenseQA

deactivate

#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=evaluate_current_results
#SBATCH --mem=20GB

module purge

module load Python/3.10.4-GCCcore-11.3.0


# Create venv if not exists
if [ ! -d "thesis_venv" ]; then
  python3 -m venv thesis_venv
fi

source ./thesis_venv/bin/activate


pip install --upgrade pip
pip install --quiet -r requirements.txt

# Run the script with args
python -m src.evaluate_uncertainty --model gemma9b

deactivate
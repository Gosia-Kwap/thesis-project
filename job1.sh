#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=first_attempt
#SBATCH --mem=10GB

# Create venv if not exists
if [ ! -d "thesis_venv" ]; then
  python3 -m venv thesis_venv
fi

# Activate venv
source thesis_venv/bin/activate

module purge

module load Python/3.10.4-GCCcore-11.3.0

source ./thesis_venv/bin/activate

git pull origin main

pip install --upgrade pip
pip install -r requirements.txt

python ./src/main.py

deactivate
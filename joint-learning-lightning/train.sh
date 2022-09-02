#!/usr/bin/env bash
#SBATCH -n 20
#SBATCH --gres=gpu:3
#SBATCH --mem-per-cpu=2048
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END

CONDA_ENV=loki

# activate conda env
eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV

python3 JointLearningLightning.py

#!/bin/bash
#SBATCH --job-name=predict-follicles
#SBATCH --output=../slurm/slurm-%j.out
#SBATCH --error=../slurm/slurm-%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=gpushort

# Load CUDA modules for GPU access
module load cuda12.9/toolkit/12.9.1
module load cuda12.9/blas/12.9.1
module load cuda12.9/fft/12.9.1 

# Run prediction script
uv run scripts/predict_follicles.py
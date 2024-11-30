#!/bin/bash
#SBATCH --partition=gpu                # Request the GPU partition
#SBATCH --gres=gpu:1                   # Request 1 GPU
#SBATCH --ntasks=1                     # Number of tasks (usually 1 for PyTorch)
#SBATCH --cpus-per-task=4              # Number of CPU cores per task
#SBATCH --mem=16G                      # Memory per node
#SBATCH --time=04:00:00                # Maximum runtime (hh:mm:ss)
#SBATCH --job-name=image_translation   # Job name
#SBATCH --output=logs/output_%j.log    # Save the output to a log file (unique ID: %j)
#SBATCH --error=logs/error_%j.log      # Save errors to a log file (unique ID: %j)

# Load necessary modules (adjust to your cluster's setup)
# module load python/3.10                # 
# module load cuda/11.8                  # CUDA version (we need to ensure compatibility)

# Maybe activate Conda environment?
source activate image_translation               

# Navigate to the project directory
cd image-to-image-translation-with-cGAN  


# Run the script
python scripts/train_apply.py

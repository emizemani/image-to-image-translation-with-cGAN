#!/bin/bash -l

#SBATCH --job-name=image_translation
#SBATCH --partition=gpu-teaching-2d # Run on the 2h GPU runtime partition, also 5h, 2d and 7d available
#SBATCH --gpus=1
current_datetime=$(date +"%Y-%m-%d_%H-%M-%S")
#SBATCH --output=/home/pml12/script/r-${current_datetime}/output.txt
#SBATCH --error=/home/pml12/script/r-${current_datetime}/error.txt
#SBATCH --chdir=/home/pml12/MS2/image-to-image-translation-with-cGAN

container=pml.sif

apptainer run --nv $container python -u src/train_apply.py

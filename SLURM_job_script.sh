#!/bin/bash -l

#SBATCH --job-name=image_translation
#SBATCH --partition=gpu-teaching-2h # Run on the 2h GPU runtime partition, also 5h, 2d and 7d availabl
#SBATCH --gpus-per-node=40gb:1 # One A100 with 40GB
#SBATCH --ntasks-per-node=4 # 4 CPU threads
#SBATCH --output=/home/pml12/script/output.txt
#SBATCH --error=/home/pml12/script/error.txt
#SBATCH --chdir=/home/pml12
#SBATCH --array=1 # run one times

container=/MS2/image-to-image-translation-with-cGAN/pml.sif

apptainer run --nv $container \
    python src/train_apply.py

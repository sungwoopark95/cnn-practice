#!/bin/bash

#SBATCH --job-name=dl-experiment
#SBATCH --nodes=1             
#SBATCH --gres=gpu:4          
#SBATCH --time=0-12:00:00  
#SBATCH --mem=64000MB

source /home/${USER}/.bashrc
conda activate main

srun python3 /home/sungwoopark/dl-practice/vision/main.py --dataset cifar100 --bs 512 --wandb convnet-exp --name vgg19 --img_size 224 --drop_cnn 0 --drop_fc 0.5 --save_plot --tqdm --num_workers 4 --lr 5e-4
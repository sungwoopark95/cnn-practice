#!/bin/bash

#SBATCH --job-name=jupyter    # 작업창에 뜨는 이름
#SBATCH --nodes=1              # 몇개 노드 받을건지
#SBATCH --gres=gpu:1           # gpu 1개
#SBATCH --time=0-12:00:00  # 12 hours timelimit
#SBATCH --mem=64000MB          # RAM

source /home/${USER}/.bashrc
conda activate main

srun python3 /home/sungwoopark/dl-practice/vision/main.py --dataset cifar100 --bs 256 --wandb convnet-exp --name ZFNet --img_size 224 --drop_cnn 0 --drop_fc 0.5 --save_plot
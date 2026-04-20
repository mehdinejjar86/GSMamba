#!/bin/bash
#SBATCH --job-name=gsMamba
#SBATCH --partition=gpu
#SBATCH --output=gs_logs/gsmamba_%j.out
#SBATCH --error=gs_logs/gsmamba_%j.err
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --gres=gpu:a40:1
#SBATCH --time=7-00:00:00

mkdir -p gs_logs_eval

source /home/groups/ChangLab/govindsa/miniconda3/etc/profile.d/conda.sh
conda activate vfimamba
python eval.py \
    --checkpoint /home/exacloud/gscratch/ChangLab/govindsa/GSMamba/outputs/gsmamba_20260314_180221/best.pth \
    --dataset x4k \
    --x4k_root /home/exacloud/gscratch/ChangLab/govindsa/Extreme/test \
    --model gsmamba_large


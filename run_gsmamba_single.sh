#!/bin/bash
#SBATCH --job-name=gsMamba
#SBATCH --partition=gpu
#SBATCH --output=gs_logs/gsmamba_%j.out
#SBATCH --error=gs_logs/gsmamba_%j.err
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --gres=gpu:a40:1
#SBATCH --time=24:00:00

mkdir -p gs_logs

source /home/groups/ChangLab/govindsa/miniconda3/etc/profile.d/conda.sh
conda activate vfimamba

python train.py \
    --exp_name gs_mamba_mixed \
    --mode mixed \
    --vimeo_root /home/groups/ChangLab/govindsa/confocal_project/TEMPO/datasets_atlas_specimen_ch0_ch1_80_20_split/Video_vimeo_triplet \
    --x4k_root /home/groups/ChangLab/govindsa/confocal_project/TEMPO/datasets_atlas_specimen_ch0_ch1_80_20_split/Extreme \
    --x4k-steps 7 15 31 \
    --x4k-n-frames 4 3 2 \
    --crop_size 256 \
    --x4k_crop_size 256 \
    --image_size 256 256 \
    --batch_size 4 \
    --epochs 10 \
    --lr 2e-4 \
    --no_amp \
    --eval_full_every 10

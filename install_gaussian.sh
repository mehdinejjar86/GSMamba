#!/bin/bash
#SBATCH --job-name=installGaus
#SBATCH --partition=gpu
#SBATCH --output=gs_logs/install_gauss_%j.out
#SBATCH --error=gs_logs/install_gauss_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --gres=gpu:a40:1
#SBATCH --time=1:00:00

source /home/groups/ChangLab/govindsa/miniconda3/etc/profile.d/conda.sh
conda activate vfimamba

cd /home/groups/ChangLab/govindsa/confocal_project/GSMamba/gaussian-splatting/submodules/diff-gaussian-rasterization
pip install .

# Verify
python -c "from diff_gaussian_rasterization import GaussianRasterizer; print('Success!')"

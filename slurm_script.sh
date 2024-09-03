#!/bin/bash
time=0-06:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --job-name="Brook Job"
#SBATCH --mail-user=s4842338@student.uq.edu.au
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH —output=output_dir/%j.out
#module load tensorflow/1.9.0
#export PYTHONPATH=~/.local/lib/python3.6/site-packages/
export PYTHONPATH=/home/Student/s4842338/.local/lib/miniconda3/envs/segment_anything_1/lib/python3.10/site-packages/
#export LD_PRELOAD=/home/Student/s4842338/.local/lib/miniconda3/envs/segment_anything_1/lib/python3.10/site-packages/nvidia/cudnn/lib/libcudnn.so.9
#cd ../checkpoints
#./download_ckpts.sh
lspci | grep -i nvidia
conda activate segment_anything_1
#python run_sam.py
python --version
pip --version
python points_run_model.py

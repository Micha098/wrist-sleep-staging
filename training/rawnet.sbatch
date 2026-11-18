#!/bin/bash
# Raw-signal wrist network (wrist/rawnet.py), one family fold per array task.
#   pilot : sbatch --array=0 --time=00:45:00 --export=ALL,RAWNET_MAX_EPOCHS=2 rawnet.sbatch
#   full  : sbatch --array=0-4 rawnet.sbatch                 (RAWNET_INPUT=both | raw | minute)
#SBATCH --job-name=wrist_rawnet
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=a100-80gb|h100  # torch 2.6 (cu12.5) has no Blackwell kernels; avoids MIG slices
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/wrist_rawnet_%A_%a.out

module purge
module load python cudnn/8.9.7.29-12  # the python torch build lacks libcudnn_cnn_train.so.8
export OMP_NUM_THREADS=8
export SOP_PROJ=${SOP_PROJ:?set SOP_PROJ to the project data folder}
export WRIST_LABELS=${WRIST_LABELS:-quality}
cd "${SLURM_SUBMIT_DIR:-.}"  # submit from the training/ folder of this repo
nvidia-smi --query-gpu=name,memory.total --format=csv
env | grep -E '^(WRIST|RAWNET)_' | sort
python -W ignore rawnet.py

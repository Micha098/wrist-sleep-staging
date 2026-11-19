#!/bin/bash
# Pack raw skin temperature per epoch (prep/pack_temp.py) for wrist/rawnet.py RAWNET_INPUT=raw_temp. CPU only.
#SBATCH --job-name=wrist_temp_pack
#SBATCH --partition=gen
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/mnt/ceph/users/mhacohen/Sleep_Onset_prediction_model/cache/logs/wrist_temp_pack_%j.out

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd /mnt/home/mhacohen/Sleep_Onset_prediction_model/prep
/mnt/home/mhacohen/miniconda3/envs/sleep/bin/python pack_temp.py --workers 16

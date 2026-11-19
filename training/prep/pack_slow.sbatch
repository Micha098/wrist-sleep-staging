#!/bin/bash
# Pack a slow raw signal per epoch (prep/pack_slow.py) for wrist/rawnet.py RAWNET_EXTRA. CPU only.
#SBATCH --job-name=wrist_slow_pack
#SBATCH --partition=gen
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/mnt/ceph/users/mhacohen/Sleep_Onset_prediction_model/cache/logs/wrist_slow_pack_%j.out

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd /mnt/home/mhacohen/Sleep_Onset_prediction_model/prep
/mnt/home/mhacohen/miniconda3/envs/sleep/bin/python pack_slow.py --signal ${SIGNAL:?set SIGNAL=temp or eda} --workers 16

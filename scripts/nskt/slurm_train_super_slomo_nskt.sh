#!/bin/bash
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -J dm_interp_nskt_superslomo
#SBATCH --mail-user=pren@lbl.gov
#SBATCH --mail-type=all
#SBATCH -t 24:00:00
#SBATCH -A m4633
#SBATCH --gpu-bind=none

module load python
conda activate /pscratch/sd/p/puren93/conda_env/genai
cd ../../

# run nskt
python train_super_slomo.py \
    --model SuperSloMo \
    --run_name SuperSloMo \
    --data_name nskt \
    --optimizer 'adam' \
    --batch_size 64 \
    --learning_rate 6e-5 \
    --epochs 1500 \
    --sampling_freq 5 \
    --patch_size 256 \
    --stride 32 \
    --total_interp_steps_train 20 \
    --checkpoint_path '' \
    --is_T_fixed True \
#!/bin/bash
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -J dm_interp_sst
#SBATCH --mail-user=pren@lbl.gov
#SBATCH --mail-type=all
#SBATCH -t 24:00:00
#SBATCH -A m4633
#SBATCH --gpu-bind=none

module load python
conda activate /pscratch/sd/p/puren93/conda_env/genai
cd ../../

# run sea temperature, flex; around 40 mins per epoch
python train_unet.py \
    --run_name FLEX \
    --data_name 'sea_temp' \
    --model FLEX \
    --flex_model_size "small" \
    --flex_mlp_ratio 2 \
    --sampling_freq 5 \
    --optimizer 'lion' \
    --epochs 50 \
    --batch_size 16 \
    --learning_rate 1e-5 \
    --checkpoint_path '' \
    --total_interp_steps_train 10 \
    --patch_size 128 \
    --stride 64 \
    --is_T_fixed False \
    --scratch_dir '/global/cfs/projectdirs/m4633/puren/interp_dm/sea_temp/'


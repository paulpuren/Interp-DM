#!/bin/bash
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -J dm_interp
#SBATCH --mail-user=pren@lbl.gov
#SBATCH --mail-type=all
#SBATCH -t 24:00:00
#SBATCH -A m1516
#SBATCH --gpu-bind=none

module load python
conda activate /pscratch/sd/p/puren93/conda_env/genai


# python train.py \
#     --model UNetVIT \
#     --batch_size 12 \
#     --run_name UNetVIT_woT_onehot \
#     --learning_rate 0.0002 \
#     --epochs 30 \
#     --patch_size 256 \
#     --stride 128 \
#     --use_last_snapshot True \
#     --num_pred_steps 10 \

# python train.py \
#     --model FLEX \
#     --batch_size 12 \
#     --run_name FLEX_refine_adam_1e-4_mlpr2 \
#     --learning_rate 1e-4 \
#     --epochs 200 \
#     --patch_size 256 \
#     --stride 128 \
#     --use_last_snapshot True \
#     --num_pred_steps 10 \
#     --checkpoint_path ''

# run sea temperature, flex
python train.py \
    --model FLEX \
    --run_name FLEX \
    --optimizer 'adam' \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --epochs 60 \
    --patch_size 128 \
    --total_interp_steps 10 \
    --stride 64 \
    --checkpoint_path '' \
    --is_T_fixed False \
    --data_name 'sea_temp' \
    --scratch_dir '/global/cfs/projectdirs/m4633/puren/interp_dm/sea_temp/'
#!/bin/bash
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -J dm_interp_nskt_flex
#SBATCH --mail-user=pren@lbl.gov
#SBATCH --mail-type=all
#SBATCH -t 24:00:00
#SBATCH -A m4633
#SBATCH --gpu-bind=none

module load python
conda activate /pscratch/sd/p/puren93/conda_env/genai
cd ../../

PRETRAINED_CHECKPOINT="./checkpoints/checkpoint_Model_FLEX_small_mlp2_Data_nskt_Optim_adam_lr0.0001_epoch200_stride32_T20_TfixedFalse.pt"

# run NSKT data
python train.py \
    --run_name FLEX \
    --data_name 'nskt' \
    --model FLEX \
    --flex_model_size "small" \
    --flex_mlp_ratio 2 \
    --sampling_freq 5 \
    --optimizer 'adam' \
    --epochs 200 \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --total_interp_steps_train 20 \
    --patch_size 256 \
    --stride 32 \
    --checkpoint_path $PRETRAINED_CHECKPOINT \
    --is_T_fixed False \

# --checkpoint_path '' \

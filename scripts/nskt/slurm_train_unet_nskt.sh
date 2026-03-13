#!/bin/bash
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -J dm_interp_nskt_unet
#SBATCH --mail-user=pren@lbl.gov
#SBATCH --mail-type=all
#SBATCH -t 24:00:00
#SBATCH -A m4633
#SBATCH --gpu-bind=none

module load python
conda activate /pscratch/sd/p/puren93/conda_env/genai
cd ../../

PRETRAINED_CHECKPOINT="./checkpoints/checkpoint_Model_UNet_Data_nskt_Optim_adam_lr0.0001_epoch200_stride32_T20_TfixedFalse.pt"

# run nskt
python train_unet.py \
    --model UNet \
    --run_name UNet \
    --data_name 'nskt' \
    --sampling_freq 10 \
    --optimizer 'adam' \
    --epochs 200 \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --total_interp_steps_train 20 \
    --is_T_fixed True \
    --patch_size 256 \
    --stride 32 \
    --scratch_dir '/global/cfs/cdirs/m4633/foundationmodel/nskt_tensor/' \
    --checkpoint_path '' 

# 
# --checkpoint_path $PRETRAINED_CHECKPOINT \
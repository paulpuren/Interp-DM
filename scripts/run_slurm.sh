#!/bin/bash
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -J dm_interp
#SBATCH --mail-user=pren@lbl.gov
#SBATCH --mail-type=all
#SBATCH -t 18:00:00
#SBATCH -A m1516
#SBATCH --gpu-bind=none

module load python
conda activate /pscratch/sd/p/puren93/conda_env/genai
cd ..

# run nskt
python train.py \
    --model FLEX \
    --run_name FLEX \
    --data_name 'nskt' \
    --sampling_freq 20 \
    --optimizer 'lion' \
    --epochs 300 \
    --batch_size 32 \
    --learning_rate 1e-5 \
    --checkpoint_path '' \
    --total_interp_steps 20 \
    --is_T_fixed False \
    --patch_size 128 \
    --stride 64 \
    --scratch_dir '/global/cfs/cdirs/m4633/foundationmodel/nskt_tensor/'

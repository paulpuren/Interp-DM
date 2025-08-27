# salloc -N 1 -C gpu -q interactive -t 04:00:00 -G 4 -A m1516

module load python
conda activate /pscratch/sd/p/puren93/miniforge3.bak-20250811-1035/envs/sft

# # run nskt
# python train.py \
#     --model FLEX \
#     --run_name FLEX \
#     --optimizer 'adam' \
#     --batch_size 12 \
#     --learning_rate 1e-4 \
#     --epochs 50 \
#     --patch_size 128 \
#     --stride 64 \
#     --checkpoint_path '' \
#     --is_T_fixed False \

# run shanghai
python train.py \
    --model FLEX \
    --run_name FLEX \
    --optimizer 'adam' \
    --batch_size 32 \
    --learning_rate 3e-4 \
    --epochs 300 \
    --patch_size 128 \
    --total_interp_steps 5 \
    --stride 64 \
    --checkpoint_path '' \
    --is_T_fixed False \
    --data_name 'shanghai' \
    --scratch_dir '/global/cfs/cdirs/m4633/puren/interp_dm/shanghai/shanghai.h5'


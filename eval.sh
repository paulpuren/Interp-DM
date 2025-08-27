# salloc -N 1 -C gpu -q interactive -t 01:00:00 -G 4 -A m1516

module load python
conda activate /pscratch/sd/p/puren93/miniforge3.bak-20250811-1035/envs/sft

python eval_shanghai.py \
    --model FLEX \
    --data_name 'shanghai' \
    --re_num_id -1 \
    --batch_size 24 \
    --total_interp_steps 18 \
    --total_interp_steps_train 10 \
    --optimizer 'adam' \
    --learning_rate 3e-4 \
    --epochs 300 \
    --patch_size 128 \
    --stride 64 \
    --is_T_fixed False \
    --scratch_dir '/global/cfs/cdirs/m4633/puren/interp_dm/shanghai/shanghai.h5'

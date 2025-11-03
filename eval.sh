# salloc -N 1 -C gpu -q interactive -t 01:00:00 -G 4 -A m1516

module load python
conda activate /pscratch/sd/p/puren93/conda_env/genai

python eval_shanghai.py \
    --model FLEX \
    --data_name 'shanghai' \
    --re_num_id -1 \
    --batch_size 16 \
    --total_interp_steps 10 \
    --total_interp_steps_train 20 \
    --optimizer 'adam' \
    --learning_rate 1e-4 \
    --epochs 300 \
    --patch_size 128 \
    --stride 64 \
    --is_T_fixed False \
    --scratch_dir '/global/cfs/cdirs/m4633/puren/interp_dm/shanghai/shanghai.h5'

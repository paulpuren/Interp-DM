# salloc -N 1 -C gpu -q interactive -t 01:00:00 -G 4 -A m1516

module load python
conda activate /pscratch/sd/p/puren93/miniforge3.bak-20250811-1035/envs/sft

python eval.py \
    --model FLEX \
    --re_num_id -1 \
    --batch_size 12 \
    --total_interp_step 5 \
    --optimizer 'adam' \
    --learning_rate 1e-4 \
    --epochs 50 \
    --is_T_fixed True \


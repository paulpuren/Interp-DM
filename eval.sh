# salloc -N 1 -C gpu -q interactive -t 04:00:00 -G 4 -A m1516

module load python
conda activate /pscratch/sd/p/puren93/miniforge3/envs/sft

python eval.py \
    --model FLEX \
    --batch_size 12 \
    --total_interp_step 10 \
    --optimizer 'adam' \
    --learning_rate 1e-4 \
    --epochs 50 \
    --is_T_fixed True \


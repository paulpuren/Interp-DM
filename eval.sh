# salloc -N 1 -C gpu -q interactive -t 04:00:00 -G 4 -A m1516

module load python
conda activate /pscratch/sd/p/puren93/miniforge3/envs/sft

python eval.py \
    --model FLEX_refine_adam_1e-4_mlpr2 \
    --batch_size 12 \
    --num_pred_steps 10 \
    --Reynolds_number 32000 \
    --if_normalize True \
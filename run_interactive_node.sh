# salloc -N 1 -C gpu -q interactive -t 04:00:00 -G 4 -A m1516

#export CUDA_VISIBLE_DEVICES=6,7;  python train.py --model UNetVIT --batch-size 18 --run-name UNetVIT --learning-rate 0.0002 --epochs 300

module load python
conda activate /pscratch/sd/p/puren93/miniforge3/envs/sft

python train.py \
    --model FLEX \
    --batch_size 12 \
    --run_name FLEX_refine_adam_1e-4_mlpr2_norm \
    --learning_rate 1e-4 \
    --epochs 50 \
    --patch_size 256 \
    --stride 128 \
    --use_last_snapshot True \
    --if_normalize True \
    --num_pred_steps 10 \
    --checkpoint_path ''


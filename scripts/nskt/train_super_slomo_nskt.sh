# salloc -N 1 -C gpu -q interactive -t 04:00:00 -G 4 -A m4633

module load python
conda activate /pscratch/sd/p/puren93/conda_env/genai

cd ../../

# run nskt
python train_super_slomo.py \
    --model SuperSloMo \
    --run_name SuperSloMo \
    --data_name nskt \
    --optimizer 'adam' \
    --batch_size 64 \
    --learning_rate 6e-5 \
    --epochs 200 \
    --sampling_freq 5 \
    --patch_size 256 \
    --stride 32 \
    --total_interp_steps_train 20 \
    --checkpoint_path '' \
    --is_T_fixed True \


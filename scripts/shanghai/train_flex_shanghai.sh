# salloc -N 1 -C gpu -q interactive -t 04:00:00 -G 4 -A m5262

module load python
conda activate /pscratch/sd/p/puren93/conda_env/genai
cd ../../

PRETRAINED_PATH="checkpoints/checkpoint_Model_FLEX_medium_mlp4_Data_shanghai_Optim_adam_lr0.0001_epoch400_stride32_T8_TfixedFalse.pt"


# run shanghai radar data
python train.py \
    --run_name FLEX \
    --data_name 'shanghai' \
    --model FLEX \
    --flex_model_size "small" \
    --flex_mlp_ratio 2 \
    --sampling_freq 10 \
    --optimizer 'adam' \
    --epochs 400 \
    --batch_size 12 \
    --learning_rate 1e-4 \
    --total_interp_steps_train 8 \
    --is_T_fixed False \
    --patch_size 128 \
    --stride 32 \
    --scratch_dir '/global/cfs/cdirs/m4633/puren/interp_dm/shanghai/shanghai.h5' \
    --checkpoint_path ""

# --checkpoint_path $PRETRAINED_PATH \
# --checkpoint_path "" \
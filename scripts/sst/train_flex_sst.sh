# salloc -N 1 -C gpu -q interactive -t 04:00:00 -G 4 -A m4633
# salloc -N 1 -C gpu -q interactive -t 04:00:00 -G 4 -A m5262

module load python
conda activate /pscratch/sd/p/puren93/conda_env/genai
cd ../../

PRETRAINED_PATH="checkpoints/checkpoint_Model_FLEX_small_mlp2_Data_sea_temp_Optim_adam_lr0.0001_epoch30_stride64_T10_TfixedFalse.pt"

# run sea temperature, flex; around 40 mins per epoch
python train.py \
    --run_name FLEX \
    --data_name 'sea_temp' \
    --model FLEX \
    --flex_model_size "small" \
    --flex_mlp_ratio 2 \
    --sampling_freq 5 \
    --optimizer 'adam' \
    --epochs 30 \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --checkpoint_path $PRETRAINED_PATH \
    --total_interp_steps_train 10 \
    --patch_size 128 \
    --stride 64 \
    --is_T_fixed False \
    --scratch_dir '/global/cfs/projectdirs/m4633/puren/interp_dm/sea_temp/'

#    --checkpoint_path '' \
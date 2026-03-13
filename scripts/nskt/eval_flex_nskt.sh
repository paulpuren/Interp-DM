# salloc -N 1 -C gpu -q interactive -t 04:00:00 -G 4 -A m4633

module load python
conda activate /pscratch/sd/p/puren93/conda_env/genai
cd ../../

# # run nskt
# python eval.py \
#     --model FLEX \
#     --data_name 'nskt' \
#     --re_id 1 \
#     --optimizer 'lion' \
#     --epochs 300 \
#     --batch_size 32 \
#     --learning_rate 1e-5 \
#     --total_interp_steps 10 \
#     --trained_total_interp_steps 20 \
#     --is_T_fixed False \
#     --target_resolution 128 \
#     --stride 64 \
#     --scratch_dir '/global/cfs/cdirs/m4633/foundationmodel/nskt_tensor/'

PRETRAINED_CHECKPOINT="./checkpoints/checkpoint_Model_FLEX_small_mlp2_Data_nskt_Optim_adam_lr0.0001_epoch200_stride32_T20_TfixedFalse.pt"

# run nskt
python eval.py \
    --model FLEX \
    --data_name 'nskt' \
    --re_id 0 \
    --optimizer 'adam' \
    --flex_model_size "small" \
    --flex_mlp_ratio 2 \
    --epochs 200 \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --total_interp_steps 16 \
    --total_interp_steps_train 20 \
    --is_T_fixed False \
    --patch_size 256 \
    --stride 32 \
    --checkpoint_path $PRETRAINED_CHECKPOINT \
    --scratch_dir '/global/cfs/cdirs/m4633/foundationmodel/nskt_tensor/'
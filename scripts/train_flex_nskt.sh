# salloc -N 1 -C gpu -q interactive -t 04:00:00 -G 4 -A m1516

# sh run_interactive_node.sh

module load python
conda activate /pscratch/sd/p/puren93/conda_env/genai
cd ..

# run nskt
python train.py \
    --model FLEX \
    --run_name FLEX \
    --data_name 'nskt' \
    --sampling_freq 20 \
    --optimizer 'lion' \
    --epochs 300 \
    --batch_size 32 \
    --learning_rate 1e-5 \
    --checkpoint_path '' \
    --total_interp_steps 20 \
    --is_T_fixed False \
    --patch_size 128 \
    --stride 64 \
    --scratch_dir '/global/cfs/cdirs/m4633/foundationmodel/nskt_tensor/'


# # run shanghai, flex
# python train.py \
#     --model FLEX \
#     --run_name FLEX \
#     --optimizer 'adam' \
#     --batch_size 32 \
#     --learning_rate 1e-4 \
#     --epochs 400 \
#     --patch_size 128 \
#     --total_interp_steps 10 \
#     --stride 64 \
#     --checkpoint_path '' \
#     --is_T_fixed False \
#     --data_name 'shanghai' \
#     --scratch_dir '/global/cfs/cdirs/m4633/puren/interp_dm/shanghai/shanghai.h5'

# # run shanghai, unet
# python train_unet.py \
#     --model UNet \
#     --run_name UNet \
#     --optimizer 'adam' \
#     --batch_size 32 \
#     --learning_rate 1e-4 \
#     --epochs 300 \
#     --patch_size 128 \
#     --total_interp_steps 20 \
#     --stride 64 \
#     --checkpoint_path '' \
#     --is_T_fixed False \
#     --data_name 'shanghai' \
#     --scratch_dir '/global/cfs/cdirs/m4633/puren/interp_dm/shanghai/shanghai.h5'

# # run sea temperature, flex
# python train.py \
#     --model FLEX \
#     --run_name FLEX \
#     --optimizer 'adam' \
#     --batch_size 32 \
#     --learning_rate 1e-4 \
#     --epochs 30 \
#     --patch_size 128 \
#     --total_interp_steps 10 \
#     --stride 64 \
#     --checkpoint_path '' \
#     --is_T_fixed False \
#     --data_name 'sea_temp' \
#     --scratch_dir '/global/cfs/projectdirs/m4633/puren/interp_dm/sea_temp/'

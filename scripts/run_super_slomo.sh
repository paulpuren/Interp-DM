# salloc -N 1 -C gpu -q interactive -t 04:00:00 -G 4 -A m1516

# sh run_interactive_node.sh

module load python
conda activate /pscratch/sd/p/puren93/conda_env/genai

cd ..

# run nskt
python train_super_slomo.py \
    --model SuperSloMo \
    --run_name SuperSloMo \
    --optimizer 'adam' \
    --batch_size 64 \
    --learning_rate 1e-4 \
    --epochs 200 \
    --sampling_freq 5 \
    --patch_size 256 \
    --stride 64 \
    --total_interp_steps 10 \
    --checkpoint_path '' \
    --is_T_fixed True \

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

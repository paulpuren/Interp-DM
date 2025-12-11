import argparse
def get_args():
    parser = argparse.ArgumentParser(
        description = "FLEX for Temporal Interpolation"
    )
    # general parameters
    parser.add_argument(
        "--run_name", 
        type = str, 
        default = 'run1', 
        help = "Name of the current run."
    )
    parser.add_argument(
        "--data_name", 
        type = str, 
        default = 'nskt', 
        help = "Name of the dataset."
    )
    parser.add_argument(
        "--sampling_freq", 
        default = 10, 
        type = int, 
        help = "How often to save a snapshot"
    )
    # Training parameters
    parser.add_argument(
        "--optimizer", 
        type = str, 
        default = "adam", 
        help = "Optimizer: adam or lion"
    )
    parser.add_argument(
        "--epochs", 
        default = 200, 
        type = int, 
        help = "Total epochs to train the model"
    )
    parser.add_argument(
        "--batch_size", 
        default = 16, 
        type = int, 
        help = "Input batch size on each device (default: 32)"
    )
    parser.add_argument(
        "--learning_rate", 
        default = 2e-4, 
        type = float, 
        help = 'learning rate'
    ) # 1e-4 for adam; 1e-5 for lion
    parser.add_argument(
        "--checkpoint_path", 
        default = "", 
        type = str, 
        help = "for reloading checkpoint and keep training"
    )
    # dataset parameters
    parser.add_argument(
        "--total_interp_steps", 
        default=1, 
        type=int, 
        help='total interpolation steps to condition on'
    )
    parser.add_argument(
        "--is_T_fixed", 
        default = True,
        type = lambda x: (str(x).lower() == 'true'), 
        help = "fix or change T in training."
    )
    parser.add_argument(
        "--patch_size", 
        default = 256, 
        type = int, 
        help = "Patch size for the datasets"
    )
    parser.add_argument(
        "--stride", 
        default = 128, 
        type = int, 
        help = "Stride for the datasets"
    )
    parser.add_argument(
        "--scratch_dir",
        default = "/global/cfs/cdirs/m4633/foundationmodel/nskt_tensor/", 
        type = str, 
        help = "Directory for the dataset"
    )
    # Diffusion parameters
    parser.add_argument(
        "--prediction_type", 
        type = str, 
        default = 'v', 
        help = "Quantity to predict during training."
    )
    parser.add_argument(
        "--sampler", 
        type = str, 
        default = 'ddim', 
        help = "Sampler to use to generate images"
    )    
    parser.add_argument(
        "--time_steps", 
        type = int, 
        default = 10, 
        help = "Diffusion time steps for sampling"
    )    
    # model parameters
    parser.add_argument(
        "--model", 
        type = str, 
        default = 'FLEX', 
        help = "model"
    )    
    # U-Net parameters
    parser.add_argument(
        "--base_width", 
        type = int, 
        default = 128, 
        help = "Basewidth of U-Net"
    )    
    return parser.parse_args()
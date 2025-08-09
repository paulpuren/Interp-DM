import os
import torch
import numpy as np
# from src.unet import UNet
from src.flex import FLEX
from src.diffusion_model import DiffusionModel
from src.helper import *
from torch.utils.data import Dataset, DataLoader
from torch_ema import ExponentialMovingAverage
import scipy.stats
import h5py
import torch.nn as nn

# Reynolds numbers for the datasets
RE_list = [
    600, 1000, 2000, 4000, 8000, 
    12000, 16000, 24000, 32000, 36000
]
class NSTK_FC(torch.utils.data.Dataset):
    def __init__(
            self,
            total_interp_steps = 1,
            re_num_id = -1,
            patch_size = 256,
            stride = 256,
            scratch_dir = './'
        ):
        super(NSTK_FC, self).__init__()

        # get the data path
        seed = '3407'
        self.paths = [
            os.path.join(scratch_dir, f'600_2048_2048_seed_{seed}.h5'),
            os.path.join(scratch_dir, f'1000_2048_2048_seed_{seed}.h5'),
            os.path.join(scratch_dir, f'2000_2048_2048_seed_{seed}.h5'),
            os.path.join(scratch_dir, f'4000_2048_2048_seed_{seed}.h5'),
            os.path.join(scratch_dir, f'8000_2048_2048_seed_{seed}.h5'),
            os.path.join(scratch_dir, f'16000_2048_2048_seed_{seed}.h5'),
            os.path.join(scratch_dir, f'32000_2048_2048_seed_{seed}.h5'),
            os.path.join(scratch_dir, f'36000_2048_2048_seed_{seed}.h5')
        ]
        self.re_num_id = re_num_id
        self.patch_size = patch_size
        self.stride = stride
        self.total_interp_steps = total_interp_steps

        with h5py.File(self.paths[0], 'r') as f:
            self.data_shape = f['w'].shape
            print(self.data_shape)

        self.max_row = (self.data_shape[1] - self.patch_size) // self.stride + 1
        self.max_col = (self.data_shape[2] - self.patch_size) // self.stride + 1 

        self.num_patches_per_image = (
            (self.data_shape[1] - self.patch_size) // self.stride + 1) * \
            ((self.data_shape[2] - self.patch_size) // self.stride + 1)
                                     
        print(f'Number of patches per snapshot: {self.num_patches_per_image}')

    def open_hdf5(self):
        self.datasets = [h5py.File(path, 'r')['w'] for path in self.paths]

    def __getitem__(self, time_index):
        
        if not hasattr(self, 'datasets'):
            self.open_hdf5()
        
        # Randomly select a dataset and Reynolds number
        # dataset_id = -1 #use just Re=16k

        # specific embedding for different reynolds numbers
        # kind of normalization on reynolds number
        reynolds_number = RE_list[self.re_num_id] ** (1 / 4) / 14 
        reynolds_number = reynolds_number if np.random.uniform() < 0.9 else 0. 

        # Randomly choose between datasets for variation
        dataset = self.datasets[self.re_num_id]

        # Select a time index for intial state
        time_index = time_index // 17  # (should be less than 1497)

        # Randomly select a patch
        row_start = np.random.randint(0, self.max_row) * self.stride
        col_start = np.random.randint(0, self.max_col) * self.stride

        # extract the input patch
        condition_start = torch.from_numpy(
            dataset[
                time_index, 
                row_start : (row_start + self.patch_size), 
                col_start : (col_start + self.patch_size)
            ]
        ).float().unsqueeze(0)
        condition_end = torch.from_numpy(
                dataset[
                    time_index + (self.total_interp_steps + 1), 
                    row_start:(row_start + self.patch_size), 
                    col_start:(col_start + self.patch_size)
                ]
            ).float().unsqueeze(0)
        inputs = [condition_start, condition_end]
    
        # create a list to hold target patches
        targets = []
        for i in range(1, (self.total_interp_steps + 1)):
            snapshot = torch.from_numpy(
                dataset[
                    time_index + i, 
                    row_start : (row_start + self.patch_size), 
                    col_start : (col_start + self.patch_size)
                ]
            ).float().unsqueeze(0)
            targets.append(snapshot)

        # extract physical parameters
        cond_params = [
            torch.tensor(self.total_interp_steps, dtype = torch.float32), 
            torch.tensor(reynolds_number)
        ]
        return inputs, targets, cond_params

    def __len__(self):
        return 50
        # return 25000
        # return self.num_patches_per_image * 70


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description = 'Minimalistic Diffusion Model for Super-resolution'
    )
    parser.add_argument(
        '--re_num_id', 
        default = -1, 
        type = int,
        help = 'reynolds number id (0-7): check RE_list'
    )
    parser.add_argument(
        '--batch_size', 
        default = 128, 
        type = int,
        help = 'Input batch size on each device (default: 32)'
    )
    parser.add_argument(
        '--target_resolution', 
        default = 256, 
        type = int, 
        help = 'target resolution'
    )
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
        help = "Time steps for sampling"
    )
    parser.add_argument(
        '--total_interp_steps', 
        default = 1, 
        type = int,
        help = 'different prediction steps to condition on'
    )
    parser.add_argument(
        "--base_width", 
        type = int,
        default = 64, 
        help = "Basewidth of U-Net"
    )
    parser.add_argument(
        "--model", 
        type = str, 
        default = 'UNetVIT', 
        help = "model"
    )
    parser.add_argument(
        '--scratch_dir', 
        default = '/global/cfs/cdirs/m4633/foundationmodel/nskt_tensor/', 
        type = str, 
        help = 'Directory for the dataset'
    )
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
        "--learning_rate", 
        default = 2e-4, 
        type = float, 
        help = 'learning rate'
    )
    parser.add_argument(
        "--is_T_fixed", 
        default = True,
        type = lambda x: (str(x).lower() == 'true'), 
        help = "fix or change T in training."
    )
    parser.add_argument(
        "--stride", 
        default = 128, 
        type = int, 
        help = "Stride for the datasets"
    )
    args = parser.parse_args()
    
    # load model
    print("Loading the trained model...")

    # FLEX model
    encoder, task_encoder, decoder = FLEX(
        image_size = args.target_resolution, 
        in_channels = 1, 
        out_channels = 1,
        model_size = 'small',
        mlp_ratio = 2
    )
    model = DiffusionModel(
        encoder = encoder.cuda(),
        decoder = decoder.cuda(),
        task_encoder = task_encoder.cuda(),
        diff_steps = args.time_steps, #time steps for sampling
        prediction_type = args.prediction_type,
        criterion = torch.nn.L1Loss()
    )
    ema = ExponentialMovingAverage(
        model.parameters(), 
        decay = 0.999
    )

    # model save path
    checkpoint_dir = './checkpoints'
    run_name = "Model_{}_Optim_{}_lr{}_epoch{}_stride{}_Tfixed{}".format(
            args.model,
            args.optimizer,
            args.learning_rate,
            args.epochs,
            args.stride,
            args.is_T_fixed
    )
    save_path = "{}/checkpoint_{}.pt".format(
        checkpoint_dir,
        run_name
    )
    checkpoint = torch.load(save_path, weights_only = True)
    model.load_state_dict(checkpoint["model"])
    ema.load_state_dict(checkpoint["ema"])

    # set seed
    seed = 0
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)

    model.to('cuda')
    model.eval()

    # load test data
    print("Loading the test dataset...")
    test_set = NSTK_FC(
        total_interp_steps = args.total_interp_steps,
        re_num_id = args.re_num_id,
        patch_size = args.target_resolution,
        stride = 512,
        scratch_dir = args.scratch_dir
    )
    testloader = DataLoader(
        test_set,
        batch_size = args.batch_size,
        pin_memory = True,
        shuffle = False,
        # sampler=DistributedSampler(dataset),
        num_workers = 8
    )

    RFNE_error = []
    R2s = []
    print(f'Number of batches: {len(testloader)}')
    with ema.average_parameters():
        with torch.no_grad():
            model.eval()

            for i, (inputs, targets, cond_params) in enumerate(testloader):
                print(i)
                # Unpack the input tuple
                condition_start, condition_end = inputs
                condition_start = condition_start.to('cuda')
                condition_end = condition_end.to('cuda')
                
                # unpack the condition parameters
                total_interp_steps, reynolds_number = cond_params
                reynolds_number = reynolds_number.to('cuda') 
                total_interp_steps = total_interp_steps.to('cuda')

                preds = []
                len_targets = len(targets)
                for ii in range(len(targets)):
                    target_interp_step = torch.tensor(
                        (ii + 1), 
                        dtype = torch.float32
                    ).to('cuda')
                    target_interp_step = target_interp_step.repeat(
                        condition_start.shape[0]
                    )
                    predictions = model.sample(
                        condition_start.shape[0],
                        (1, args.target_resolution, args.target_resolution),
                        condition_start, 
                        condition_end, 
                        reynolds_number,
                        target_interp_step,
                        total_interp_steps,
                        'cuda'
                    )                  
                    preds.append(predictions.cpu().detach().numpy())


                for j in range(predictions.shape[0]):
                    RFNE_error_at_time_p = []
                    cc_error_at_time_p = []
                    
                    for p in range(len(targets)):

                        target = targets[p].cpu().detach().numpy()
                        prediction = preds[p]

                        # compute RFNE
                        error = (
                            np.linalg.norm(
                                prediction[j, 0, :, :] - target[j, 0, :, :]
                            ) / \
                            np.linalg.norm(
                                target[j, 0, :, :]
                            )
                        )
                        RFNE_error_at_time_p.append(error)

                        # compute correlation coefficient
                        cc = scipy.stats.pearsonr(
                            prediction[j, 0, :, :].flatten(), 
                            target[j, 0, :, :].flatten()
                        )[0]
                        cc_error_at_time_p.append(cc)

                    RFNE_error.append(RFNE_error_at_time_p)
                    R2s.append(cc_error_at_time_p)
                print(np.mean(np.vstack(R2s), axis=0 ))

                if i == 0:
                    samples = {
                        'conditioning_snapshots': condition_start.cpu().detach().numpy(),
                        'targets': targets,
                        'predictions': preds
                    }

                    if not os.path.exists("./samples"):
                        os.makedirs("./samples")
                    sample_path = "./samples/{}_RE{}_T{}".format(
                        run_name,
                        RE_list[args.re_num_id],
                        args.total_interp_steps
                    )
                    np.save(sample_path + '.npy', samples)
                    print('Generated samples saved...')

    avg_RFNE = np.mean(np.vstack(RFNE_error), axis=0)
    print(f'Average RFNE={repr(avg_RFNE)}')

    avg_R2 = np.mean(np.vstack(R2s), axis=0)
    print(f'Average Pearson correlation coefficients={repr(avg_R2)}')



# export CUDA_VISIBLE_DEVICES=7; python evaluation.py --task forecast --batch-size 32 --horizen 50 --Reynolds-number 12000

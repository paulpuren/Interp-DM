import os
import torch
import numpy as np
# from src.unet import UNet
from src.unetvit import UNetVIT
from src.ViT import UViT_Wrapper

from src.flex import FLEX
from src.diffusion_model_sr import DiffusionModel
from src.helper import *

from src.diffusion_model import GaussianDiffusionModelCast
# from src.get_data import NSTK_SR as NSTK
from torch.utils.data import Dataset, DataLoader
from torch_ema import ExponentialMovingAverage
import scipy.stats
import h5py
import torch.nn as nn


class NSTK_FC(torch.utils.data.Dataset):
    def __init__(self, 
                factor, 
                num_pred_steps = 1,
                patch_size = 256,
                stride = 256,
                # Reynolds_number = 16000,
                scratch_dir = './',
                use_last_snapshot = True):
        
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
            os.path.join(scratch_dir, f'36000_2048_2048_seed_{seed}.h5'),]

        self.RE_list = [600, 1000, 2000, 4000, 8000, 12000, 16000, 24000, 32000, 36000]

        self.factor = factor
        #self.pred_steps = pred_steps
        self.patch_size = patch_size
        self.stride = stride
        # self.Reynolds_number = Reynolds_number
        #self.horizon = horizon
        self.num_pred_steps = num_pred_steps
        self.use_last_snapshot = use_last_snapshot

        with h5py.File(self.paths[0], 'r') as f:
            self.data_shape = f['w'].shape
            print(self.data_shape)

        self.max_row = (self.data_shape[1] - self.patch_size) // self.stride + 1
        self.max_col = (self.data_shape[2] - self.patch_size) // self.stride + 1 

        self.num_patches_per_image = ((self.data_shape[1] - self.patch_size) // self.stride + 1) * \
                                     ((self.data_shape[2] - self.patch_size) // self.stride + 1)
                                     
        print(f'Number of patches per snapshot: {self.num_patches_per_image}')

    def open_hdf5(self):
        self.datasets = [h5py.File(path, 'r')['w'] for path in self.paths]

    def __getitem__(self, time_index):
        
        if not hasattr(self, 'datasets'):
            self.open_hdf5()
        
        # Randomly select a dataset and Reynolds number
        #dataset_id = np.random.randint(len(self.datasets))
        dataset_id = -1 #use just Re=16k

        # reynolds_number = self.RE_list[dataset_id]
        
        # specific embedding for different reynolds numbers
        # kind of normalization on reynolds number
        reynolds_number = self.RE_list[dataset_id]**(1/4) / 14 
        reynolds_number = reynolds_number if np.random.uniform() < 0.9 else 0. 

        # Randomly choose between datasets for variation
        dataset = self.datasets[dataset_id]

        # Select a time index for intial state
        time_index = time_index // 17  # (should be less than 1497)

        # Randomly select a patch
        row_start = np.random.randint(0, self.max_row) * self.stride
        col_start = np.random.randint(0, self.max_col) * self.stride

        # extract the input patch
        condition_start = torch.from_numpy(
            dataset[time_index, row_start:(row_start + self.patch_size), col_start:(col_start + self.patch_size)]
        ).float().unsqueeze(0)

        if self.use_last_snapshot == True:
            condition_end = torch.from_numpy(
                dataset[time_index + (self.num_pred_steps+1), row_start:(row_start + self.patch_size), col_start:(col_start + self.patch_size)]
            ).float().unsqueeze(0)
        else:
            condition_end = torch.zeros_like(condition_start)
        
        inputs = [condition_start, condition_end]
        
        # extract the target patch
        # define a random time index for the target within the range of predicted steps
        # time_index_interp = np.random.randint(0, self.num_pred_steps) + 1
        
        # targets = torch.from_numpy(
        #    dataset[time_index+time_index_interp, row_start:(row_start + self.patch_size), col_start:(col_start + self.patch_size)]
        # ).float().unsqueeze(0)
        
        # targets = torch.from_numpy(
        #     dataset[(time_index+1):(time_index+self.num_pred_steps+1), row_start:(row_start + self.patch_size), col_start:(col_start + self.patch_size)]
        # ).float().unsqueeze(0)
        
        targets = []
        for i in range(1, self.num_pred_steps+1):
            # snapshot = torch.from_numpy(dataset[time_index+1+i, row_start:(row_start + self.patch_size), col_start:(col_start + self.patch_size)]).float().unsqueeze(0)
            # targets.append(snapshot)

            snapshot = torch.from_numpy(dataset[time_index+i, row_start:(row_start + self.patch_size), col_start:(col_start + self.patch_size)]).float().unsqueeze(0)
            targets.append(snapshot)



        # extract physical parameters
        # cond_params = [torch.tensor(self.num_pred_steps), torch.tensor(reynolds_number, dtype=torch.float32), torch.tensor(time_index_interp, dtype=torch.float32)]
        cond_params = [torch.tensor(self.num_pred_steps, dtype=torch.float32), torch.tensor(reynolds_number)]
        
        
        return inputs, targets, cond_params

    def __len__(self):
        return 50
        # return 25000
        # return self.num_patches_per_image * 70


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Minimalistic Diffusion Model for Super-resolution')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Input batch size on each device (default: 32)')
    
    # parser.add_argument('--pred_steps', default=1, type=int, help='prediction step for forecasting')
    
    # parser.add_argument('--horizon', default=5, type=int, help='forecasting horizon')

    parser.add_argument('--Reynolds_number', default=12000, type=int, help='Reynolds number')
    
    parser.add_argument('--target_resolution', default=256,
                        type=int, help='target resolution')
    
    parser.add_argument('--factor', default=8, type=int, help='upsampling factor')

    parser.add_argument("--prediction_type", type=str, default='v',
                        help="Quantity to predict during training.")
    
    parser.add_argument("--sampler", type=str, default='ddim', help="Sampler to use to generate images")
    
    parser.add_argument("--time_steps", type=int, default=10,
                        help="Time steps for sampling")
    
    parser.add_argument('--num_pred_steps', default=1, type=int,
                        help='different prediction steps to condition on')

    parser.add_argument("--base_width", type=int,
                        default=64, help="Basewidth of U-Net")

    parser.add_argument("--model", type=str, default='UNetVIT', help="model")   
    
    parser.add_argument('--if_normalize', default=True, type=lambda x: (str(x).lower() == 'true'), help='whether to normalize the data or not')

    parser.add_argument('--scratch_dir', default='/global/cfs/cdirs/m4633/foundationmodel/nskt_tensor/', type=str, help='Directory for the dataset')
    
    args = parser.parse_args()
    
    # ---
    # Load Model
    # ---
    print("Loading the trained model...")
    
    # backbone = UNetVIT(
    #             image_size=args.target_resolution, 
    #             in_channels=3, 
    #             out_channels=1,
    #             base_width=args.base_width,
    #             Reynolds_number=True,
    #             num_pred_steps=args.num_pred_steps+1)

    # model = GaussianDiffusionModelCast(
    #             eps_model=backbone.cuda(),
    #             num_timesteps=args.time_steps, #time steps for sampling
    #             prediction_type=args.prediction_type)

    # FLEX model
    encoder, superres_encoder, _, decoder = FLEX(
                image_size=args.target_resolution, 
                in_channels=1, 
                out_channels=1)
        
    model = DiffusionModel(
        encoder=encoder.cuda(),
        decoder=decoder.cuda(),
        superres_encoder=superres_encoder.cuda(),
        n_T=args.time_steps, #time steps for sampling
        prediction_type=args.prediction_type,
        criterion=torch.nn.L1Loss()
    )

    # model = run_name
    save_path = f"checkpoints/checkpoint_{args.model}_step{args.num_pred_steps}_lastStepTrue_last_woT_onehot.pt"

# save_path = f"{checkpoint_dir}/checkpoint_{self.run_name}_step{self.num_pred_steps}_lastStep{self.use_last_snapshot}{name}_woT_onehot.pt"

    # # load model
    # model, ema, device = load_model(path=save_path, 
    #                             image_size=args.target_resolution, 
    #                             model_size='small',
    #                             reverse_steps = args.time_steps,
    #                             prediction_type = args.prediction_type)

    checkpoint = torch.load(save_path, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    # model.ema.load_state_dict(checkpoint["ema"])

    # model.eps_model.load_state_dict(checkpoint["model"])
    ema = ExponentialMovingAverage(model.parameters(), decay=0.999)
    ema.load_state_dict(checkpoint["ema"])

    # set seed
    seed = 0
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)

    model.to('cuda')
    model.eval()

    # ---
    # load test data
    # ---
    test_set = NSTK_FC(
                factor=args.factor,
                num_pred_steps=args.num_pred_steps,
                patch_size=args.target_resolution,
                stride=512,
                scratch_dir=args.scratch_dir)

    testloader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        pin_memory=True,
        shuffle=False,
        # sampler=DistributedSampler(dataset),
        num_workers=8)

    RFNE_error = []
    R2s = []

    print(f'Number of batches: {len(testloader)}')
    # with model.module.ema.average_parameters():
    
    with ema.average_parameters():
        with torch.no_grad():
            model.eval()

            for i, (inputs, targets, cond_params) in enumerate(testloader):

                print(i)
                
                # Unpack the input tuple
                condition_start, condition_end = inputs
                condition_start = condition_start.to('cuda')
                condition_end = condition_end.to('cuda')
                
                # targets = targets.to('cuda')
                
                # unpack the condition parameters
                prediction_step, reynolds_number = cond_params

                # prediction_step = prediction_step.to('cuda')

                reynolds_number = reynolds_number.to('cuda') 
                # if isinstance(model.module, DiffusionModel):
                reynolds_number = reynolds_number.unsqueeze(-1)

                # time_index_interp = time_index_interp.to('cuda')
                
                # conditioning_snapshots = conditioning_snapshots.to('cuda')
                # conditioning_snapshots2 = conditioning_snapshots2.to('cuda')
                # s = s.to('cuda')
                # dat_class = dat_class.to('cuda')

                preds = []
                
                # cond_snapshot1 = condition_start
                len_targets = len(targets)
                
                for ii in range(len(targets)):
                    
                    # if ii == (len(targets) - 1):
                    #     cond_snapshot2 = condition_end
                    # else:
                    # cond_snapshot2 = (len_targets - ii) / (len_targets + 1) * condition_start + (ii + 1) / (len_targets + 1) * condition_end
                    
                    # time_index_interp = ii + 1
                    # time_index_interp = torch.tensor((ii+1)/(prediction_step+1), dtype=torch.float32).to('cuda')
                    # time_index_interp = torch.tensor((ii+1), dtype=torch.float32).to('cuda')
                    time_index_interp = torch.tensor((ii+1), dtype=torch.float32).to('cuda')
                    
                    # print("time index interpolation step:", time_index_interp)
                    
                    # interp_ratio = time_index_interp / (prediction_step[0] + 1)
                    # condition_start = (interp_ratio) * condition_start + (1 - interp_ratio) * condition_end
                    # condition_end = torch.zeros_like(condition_start)
                    
                    time_index_interp = time_index_interp.repeat(condition_start.shape[0])
                    #print("time index interpolation step shape:", time_index_interp.shape)
                    
                    predictions = model.sample(condition_start.shape[0],
                                                (1, args.target_resolution, args.target_resolution),
                                                condition_start, 
                                                condition_end, 
                                                reynolds_number,
                                                time_index_interp,
                                                args.num_pred_steps,
                                                'cuda',
                                                if_normalize=args.if_normalize)
                    
                    
                    # # multiple times of random sampling
                    # predictions = []
                    # for _ in range(1): 
                    #     prediction = model.sample(
                    #         targets.shape[0],
                    #         (1, targets.shape[2], targets.shape[3]),
                    #         cond_snapshot1,
                    #         cond_snapshot2,
                    #         prediction_step,
                    #         reynolds_number,
                    #         'cuda')
                        
                    #     # prediction = model.sample(conditioning_snapshots.shape[0], 
                    #     #                             (1, args.target_resolution, args.target_resolution),
                    #     #                             conditioning_snapshots, conditioning_snapshots2, s, dat_class,
                    #     #                             'cuda')
                        
                    #     predictions.append(prediction)
                            
                    # predictions = torch.mean(torch.stack(predictions), 0)                        
                    
                    preds.append(predictions.cpu().detach().numpy())

                    # conditioning_snapshots2 = predictions
                    # cond_snapshot1 = predictions
                    
                    # if ii == (len(targets) - 1):
                    #     cond_snapshot2 = condition_end


                for j in range(predictions.shape[0]):
                    RFNE_error_at_time_p = []
                    cc_error_at_time_p = []
                    
                    for p in range(len(targets)):

                        target = targets[p].cpu().detach().numpy()
                        prediction = preds[p]

                        # compute RFNE
                        error = np.linalg.norm(
                            prediction[j, 0, :, :] - target[j, 0, :, :]) / np.linalg.norm(target[j, 0, :, :])
                        RFNE_error_at_time_p.append(error)

                        # compute correlation coef
                        cc = scipy.stats.pearsonr(
                            prediction[j, 0, :, :].flatten(), target[j, 0, :, :].flatten())[0]
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

                    np.save(
                        f'samples/samples_forecast_RE_{args.Reynolds_number}_SR_{args.sampler}_{args.time_steps}_unet_{args.base_width}_' + str(i+1) + '.npy', samples)
                    print('saved samples')

                #if i == 5:
                #    break

    avg_RFNE = np.mean(np.vstack(RFNE_error), axis=0)
    print(f'Average RFNE={repr(avg_RFNE)}')

    avg_R2 = np.mean(np.vstack(R2s), axis=0)
    print(f'Average Pearson correlation coefficients={repr(avg_R2)}')



# export CUDA_VISIBLE_DEVICES=7; python evaluation.py --task forecast --batch-size 32 --horizen 50 --Reynolds-number 12000

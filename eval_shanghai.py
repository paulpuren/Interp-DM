
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
from torchvision import transforms 
import cv2

PIXEL_SCALE = 90.0

COLOR_MAP = np.array([
    [0, 0, 0,0],
    [0, 236, 236, 255],
    [1, 160, 246, 255],
    [1, 0, 246, 255],
    [0, 239, 0, 255],
    [0, 200, 0, 255],
    [0, 144, 0, 255],
    [255, 255, 0, 255],
    [231, 192, 0, 255],
    [255, 144, 2, 255],
    [255, 0, 0, 255],
    [166, 0, 0, 255],
    [101, 0, 0, 255],
    [255, 0, 255, 255],
    [153, 85, 201, 255],
    [255, 255, 255, 255]
    ]) / 255

BOUNDS = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75, PIXEL_SCALE]
THRESHOLDS = [20, 30, 35, 40]

HMF_COLORS = np.array([
    [82, 82, 82],
    [252, 141, 89],
    [255, 255, 191],
    [145, 191, 219]
]) / 255

def cal_ssim(pred, true, data_range = 255):
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    img1 = pred.astype(np.float64)
    img2 = true.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5 : -5, 5 : -5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5 : -5, 5 : -5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5 : -5, 5 : -5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5 : -5, 5 : -5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5 : -5, 5 : -5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

class Shanghai_Eval(Dataset):
    def __init__(
            self, 
            total_interp_steps,
            data_path, 
            img_size, 
            type = 'train', 
            trans = None, 
            seq_len = -1
        ):
        super().__init__()
        self.pixel_scale = PIXEL_SCALE
        self.data_path = data_path
        self.img_size = img_size
        self.total_interp_steps = total_interp_steps

        assert type in ['train', 'test', 'val']
        self.type = type if type!='val' else 'test'
        with h5py.File(data_path,'r') as f:
            self.all_len = int(f[self.type]['all_len'][()]) 
        if trans is not None:
            self.transform = trans
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((img_size, img_size)),
                    # transforms.ToTensor(),
                    # trans.Lambda(lambda x: x/255.0),
                    # transforms.Normalize(mean=[0.5], std=[0.5]),
                    # trans.RandomCrop(data_config["img_size"]),
                ]
            )
                    
    def __len__(self):
        return self.all_len

    def sample(self):
        index = np.random.randint(0, self.all_len)
        return self.__getitem__(index)
    
    def __getitem__(self, index):

        with h5py.File(self.data_path, 'r') as f:
            # numpy array: (25, 565, 784), dtype=uint8, range(0,70)
            # 25 is seq len
            imgs = f[self.type][str(index)][()]   
            frames = torch.from_numpy(imgs).float().squeeze() 
            frames = frames / 255.0
            frames = self.transform(frames)  # [25, 128, 128]   
        # frames = frames.unsqueeze(1) # (25,1,128,128)

        # MODIFIED BY PU REN
        # define a random total pred steps
        # total_interp_steps = np.random.randint(5, 20)
        # total_interp_steps = np.random.randint(2, 5)

        # # define a random time index for the target within the range of predicted steps
        # target_interp_step = np.random.randint(0, total_interp_steps) + 1

        # extract the input patch
        condition_start = frames[0].unsqueeze(0)
        condition_end = frames[self.total_interp_steps + 1].unsqueeze(0)
        inputs = [condition_start, condition_end]
        
        # # extract the target patch
        # targets = frames[target_interp_step].unsqueeze(0)

        # create a list to hold target patches
        targets = []
        for i in range(1, (self.total_interp_steps + 1)):
            snapshot = frames[i].unsqueeze(0)
            targets.append(snapshot)

        cond_params = [
            # torch.tensor(target_interp_step, dtype=torch.float32), 
            torch.tensor(self.total_interp_steps, dtype=torch.float32),
            torch.tensor(0.0,  dtype=torch.float32),
        ]
        
        return inputs, targets, cond_params


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
        "--data_name", 
        type = str, 
        default = 'nskt', 
        help = "Name of the dataset."
    )
    parser.add_argument(
        '--batch_size', 
        default = 128, 
        type = int,
        help = 'Input batch size on each device (default: 32)'
    )
    parser.add_argument(
        '--patch_size', 
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
        help = 'total prediction steps during evaluation'
    )
    parser.add_argument(
        "--total_interp_steps_train", 
        default=1, 
        type=int, 
        help='total prediction steps during training'
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
    encoder, task_encoder, task_encoder_end, decoder = FLEX(
        image_size = args.patch_size, 
        in_channels = 1, 
        out_channels = 1,
        model_size = 'medium', # 'medium'
        mlp_ratio = 2
    )
    model = DiffusionModel(
        encoder = encoder.cuda(),
        decoder = decoder.cuda(),
        task_encoder = task_encoder.cuda(),
        task_encoder_end = task_encoder_end.cuda(),
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
    run_name = "Model2_interp_skip0.1_{}_Data_{}_Optim_{}_lr{}_epoch{}_stride{}_T{}_Tfixed{}".format(
            args.model,
            args.data_name,
            args.optimizer,
            args.learning_rate,
            args.epochs,
            args.stride,
            args.total_interp_steps_train,
            args.is_T_fixed
    )
    save_path = "{}/checkpoint_{}.pt".format(
        checkpoint_dir,
        run_name
    )
    print(f'Loading from {save_path}')
    # save_path = "./checkpoints/checkpoint_Model_FLEX_Data_shanghai_Optim_adam_lr0.0003_epoch200_stride128_TfixedFalse.pt"
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
    # test_set = NSTK_FC(
    #     total_interp_steps = args.total_interp_steps,
    #     re_num_id = args.re_num_id,
    #     patch_size = args.patch_size,
    #     stride = 512,
    #     scratch_dir = args.scratch_dir
    # )
    test_set = Shanghai_Eval(
        total_interp_steps = args.total_interp_steps,
        data_path = args.scratch_dir,
        img_size = args.patch_size, 
        type = "test",
        trans = None
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
    SSIM_lst = []
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
                # reynolds_number = reynolds_number.unsqueeze(-1)
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
                        (1, args.patch_size, args.patch_size),
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
                    ssim_at_time_p = []
                    
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

                        # compute SSIM
                        ssim = cal_ssim(
                            prediction[j, 0, :, :] * PIXEL_SCALE, 
                            target[j, 0, :, :] * PIXEL_SCALE, 
                            data_range = PIXEL_SCALE
                        )
                        ssim_at_time_p.append(ssim)

                    RFNE_error.append(RFNE_error_at_time_p)
                    R2s.append(cc_error_at_time_p)
                    SSIM_lst.append(ssim_at_time_p)
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
                        "Shanghai", #RE_list[args.re_num_id],
                        args.total_interp_steps
                    )
                    np.save(sample_path + '.npy', samples)
                    print('Generated samples saved...')

    avg_RFNE = np.mean(np.vstack(RFNE_error), axis = 0)
    print(f'Average RFNE={repr(avg_RFNE)}')

    avg_R2 = np.mean(np.vstack(R2s), axis = 0)
    print(f'Average Pearson correlation coefficients={repr(avg_R2)}')

    avg_ssim = np.mean(np.vstack(SSIM_lst), axis = 0)
    print(f'Average SSIM value={repr(avg_ssim)}')



# export CUDA_VISIBLE_DEVICES=7; python evaluation.py --task forecast --batch-size 32 --horizen 50 --Reynolds-number 12000

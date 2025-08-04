import os, sys, time
import numpy as np
import wandb
import torch
from torch_ema import ExponentialMovingAverage
from torch.utils.data import Dataset, DataLoader

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.distributed import init_process_group, destroy_process_group
from torch.distributed import init_process_group, destroy_process_group, barrier, get_rank, is_initialized, all_reduce, get_world_size
from src.lion import Lion  # optional Lion optimiser

from diffusers.optimization import get_linear_schedule_with_warmup as scheduler

from src.unet import UNet
from src.unetvit import UNetVIT
from src.ViT import UViT_Wrapper
from src.flex import FLEX
from src.diffusion_model_sr import DiffusionModel

from src.diffusion_model import GaussianDiffusionModelCast 
from src.get_data import NSKT as NSKT
from src.plotting import plot_samples

# def ddp_setup(gpu_id, world_size):
#     """
#     Initialize distributed data parallel (DDP) environment.

#     Args:
#         gpu_id (int): Unique identifier of each process (GPU).
#         world_size (int): Total number of processes.
#     """
#     os.environ["MASTER_ADDR"] = "localhost"
#     os.environ["MASTER_PORT"] = "4331"
    
#     init_process_group(backend = "nccl", rank=gpu_id, world_size=world_size)
    
#     torch.cuda.set_device(gpu_id)


def ddp_setup(local_rank, world_size):
    """
    Args:
        rank: Unique identifixer of each process
        world_size: Total number of processes
    """
    
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "3522"
        
        init_process_group(
            backend = "gloo", # nccl for multi-gpu, gloo for single-gpu
            rank = local_rank, 
            world_size = world_size)
        
        rank = local_rank
        
    else:
        init_process_group(backend="gloo", 
                           init_method='env://')
        
        #overwrite variables with correct values from env
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = get_rank()

    torch.cuda.set_device(local_rank)
    torch.backends.cudnn.benchmark = True
    return local_rank, rank

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        local_gpu_id: int,
        sampling_freq: int,
        run: wandb,
        run_name: str,
        use_last_snapshot: bool = True,
        num_pred_steps: int = 1,
        if_normalize: bool = True
    ) -> None:
        
        self.gpu_id = gpu_id
        self.local_gpu_id = local_gpu_id
        self.model = model.to(local_gpu_id)
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.sampling_freq = sampling_freq
        self.model = DDP(model, device_ids=[local_gpu_id], find_unused_parameters=True)
        self.run = run
        self.run_name = run_name
        self.use_last_snapshot = use_last_snapshot
        self.num_pred_steps = num_pred_steps
        self.if_normalize = if_normalize
    
    def _run_batch(
        self, 
        targets, 
        condition_start, 
        condition_end,  
        reynolds_number,
        time_index_interp):
        
        """
        Run a single batch for training.
        """
        
        self.optimizer.zero_grad()
        
        if isinstance(self.model.module, DiffusionModel):
            reynolds_number = reynolds_number.unsqueeze(-1)
            # time_index_interp = time_index_interp.unsqueeze(-1)
        #     loss = self.model(
        #         targets, 
        #         condition_start, 
        #         reynolds_number.unsqueeze(-1))
        # else:
        #     loss = self.model(
        #         targets, 
        #         condition_start, 
        #         condition_end, 
        #         reynolds_number, 
        #         time_index_interp)
        
        loss = self.model(
            targets, 
            condition_start, 
            condition_end, 
            reynolds_number, 
            time_index_interp,
            self.num_pred_steps,
            if_normalize=self.if_normalize)

        loss.backward()
        if isinstance(self.model.module, DiffusionModel):
            torch.nn.utils.clip_grad_norm_(self.model.module.parameters(), 1.)
        self.optimizer.step()
        self.lr_scheduler.step()
        self.model.module.ema.update()
        
        return loss.item()

    def _run_epoch(self, epoch):
        
        """
        Run a single training epoch.
        """
        
        self.train_loader.sampler.set_epoch(epoch)
        loss_values_task = []
        
        # for condition_start, condition_end, targets, prediction_step, reynolds_number in self.train_loader:
        
        for inputs, targets, cond_params in self.train_loader:
            
            # Unpack the input tuple
            condition_start, condition_end = inputs
            condition_start = condition_start.to(self.local_gpu_id)
            condition_end = condition_end.to(self.local_gpu_id)
            
            targets = targets.to(self.local_gpu_id)
            
            # unpack the condition parameters
            # prediction_step, reynolds_number, time_index_interp = cond_params
            # prediction_step = prediction_step.to(self.local_gpu_id)
            # reynolds_number = reynolds_number.to(self.local_gpu_id)
            # time_index_interp = time_index_interp.to(self.local_gpu_id)
            
            time_index_interp, reynolds_number = cond_params
            reynolds_number = reynolds_number.to(self.local_gpu_id)
            time_index_interp = time_index_interp.to(self.local_gpu_id)
            
            loss_task = self._run_batch(
                targets, 
                condition_start, 
                condition_end, 
                reynolds_number, 
                time_index_interp)
            
            loss_values_task.append(loss_task)

        self.run.log({"Task loss": np.mean(loss_values_task)})
        self.run.log({"Contrastive loss": 0})  
            
        return loss_values_task

    def _generate_samples(self, epoch):
        
        """
        Generate samples and save them at specified path.
        """
        
        sample_path = f"./train_samples_{self.run_name}_step{self.num_pred_steps}_lastStep{self.use_last_snapshot}_woT_onehot"
        os.makedirs(sample_path, exist_ok=True)

        with self.model.module.ema.average_parameters():
            
            self.model.eval()
            
            with torch.no_grad():
                self.train_loader.sampler.set_epoch(1)
                
                # condition_start, condition_end, targets, prediction_step, reynolds_number = next(iter(self.train_loader))
                
                # unpack the data
                inputs, targets, cond_params = next(iter(self.train_loader))
                condition_start, condition_end = inputs
                condition_start = condition_start.to(self.local_gpu_id)
                condition_end = condition_end.to(self.local_gpu_id)
                
                # parse the conditional physical parameters
                # prediction_step, reynolds_number, time_index_interp = cond_params 
                # prediction_step = prediction_step.to(self.local_gpu_id)
                # reynolds_number = reynolds_number.to(self.local_gpu_id)
                # time_index_interp = time_index_interp.to(self.local_gpu_id)
                time_index_interp, reynolds_number = cond_params 
                reynolds_number = reynolds_number.to(self.local_gpu_id)

                if isinstance(self.model.module, DiffusionModel):
                    reynolds_number = reynolds_number.unsqueeze(-1)
                
                time_index_interp = time_index_interp.to(self.local_gpu_id)
                print(f'Type {type(time_index_interp)}')

                samples = self.model.module.sample(
                    targets.shape[0],
                    (1, targets.shape[2], targets.shape[3]),
                    condition_start,
                    condition_end,
                    reynolds_number,
                    time_index_interp,
                    self.num_pred_steps,
                    self.local_gpu_id,
                    if_normalize=self.if_normalize
                )

        plot_samples(samples, condition_start, targets, sample_path, epoch)
        print("We use normalized data...")
        print(f"Epoch {epoch} | Generated samples saved at {sample_path}")

    def _save_checkpoint(self, epoch, name=''):
        checkpoint_dir = "./checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)

        save_path = f"{checkpoint_dir}/checkpoint_{self.run_name}_step{self.num_pred_steps}_lastStep{self.use_last_snapshot}{name}_woT_onehot.pt"
        
        save_dict = {
            'model': self.model.module.state_dict(),
            'ema': self.model.module.ema.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        
        torch.save(save_dict, save_path)
        
        if name == '':
            print(f"Epoch {epoch} | Training checkpoint saved at {save_path}")

    def train(self, max_epochs: int):
        
        print('Starting training...')
        
        self.lr_scheduler = scheduler(
            optimizer = self.optimizer,
            num_warmup_steps = len(self.train_loader) * 3, # we need only a very shot warmup phase for our data
            num_training_steps = (len(self.train_loader) * max_epochs)
        )

        best_mse = np.inf
        self.model.train()

        for epoch in range(max_epochs):
            loss_values = self._run_epoch(epoch)

            if self.local_gpu_id == 0:
                avg_loss = np.mean(loss_values)
                print(f"Epoch {epoch} | loss {avg_loss} | learning rate {self.lr_scheduler.get_last_lr()}")
                #self.run.log({"loss": avg_loss})

                # Save the last and best checkpoint
                self._save_checkpoint(epoch + 1, name = '_last')
                
                if best_mse > avg_loss:
                    self._save_checkpoint(epoch+1)
                    best_mse = avg_loss

                # Generate samples at specified intervals
                if epoch == 0 or (epoch + 1) % self.sampling_freq == 0 or (epoch + 1) == max_epochs:
                    self._generate_samples(epoch+1)
            
def load_checkpoint(save_path, model, optimizer, device):
    if not os.path.exists(save_path):
        print(f"Unable to load from {save_path}")

    # checkpoint = torch.load(save_path, map_location=device)
    # model.load_state_dict(checkpoint['model'])
    # model.ema.load_state_dict(checkpoint['ema'])
    # optimizer.load_state_dict(checkpoint['optimizer'])

    # updated version
    checkpoint = torch.load(save_path, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    ema = ExponentialMovingAverage(model.parameters(), decay=0.999)
    ema.load_state_dict(checkpoint["ema"])
    optimizer.load_state_dict(checkpoint['optimizer'])

    print(f"Loaded model from {save_path}")
    return model, ema, optimizer
          
def load_train_objs(args):
    
    train_set = NSKT(
        factor=args.factor, 
        num_pred_steps=args.num_pred_steps,
        patch_size = args.patch_size, 
        stride = args.stride,
        scratch_dir = args.scratch_dir,
        train = True,
        use_last_snapshot = args.use_last_snapshot)
    
    # if args.model == 'UNetVIT':
    #     backbone = UNetVIT(
    #         image_size=args.patch_size, 
    #         in_channels=3, 
    #         out_channels=1, 
    #         base_width=args.base_width,
    #         num_pred_steps=args.num_pred_steps+1,
    #         Reynolds_number=True)
    ema = None # placeholder for non-FLEX model
    if args.model == 'UNetVIT':
        backbone = UNetVIT(
            image_size=args.patch_size, 
            in_channels=3, 
            out_channels=1, 
            base_width=args.base_width,
            Reynolds_number=True,
            num_pred_steps=args.num_pred_steps+1)    
    
        model = GaussianDiffusionModelCast(eps_model=backbone.cuda(),
                                       num_timesteps=args.time_steps, 
                                       prediction_type = args.prediction_type)
        
        # # ===== testing ===
        #checkpoint = torch.load(
        #    "checkpoints/checkpoint_UNetVITv4.pt", weights_only=True,
        #    map_location=torch.device('cpu'))
        # optimizer.load_state_dict(checkpoint["optimizer"])
        #model.eps_model.load_state_dict(checkpoint["model"])
        #model.ema.load_state_dict(checkpoint["ema"])    
        # # ===== testing ===
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0)
        # optimizer = Lion(model.parameters(), lr=args.learning_rate)  # alt
   
    elif args.model == 'ViT':
        
        backbone = UViT_Wrapper(image_size=args.patch_size, task='flex')

        model = GaussianDiffusionModelCast(eps_model = backbone.cuda(),
                                           num_timesteps = args.time_steps, 
                                           prediction_type = args.prediction_type)
        
        no_decay = model.eps_model.no_weight_decay()

        # Separate parameters into those that will have weight decay and those that won't
        decay_params = []
        no_decay_params = []
        
        for name, param in model.named_parameters():
            if name in no_decay:
                no_decay_params.append(param)
            else:
                decay_params.append(param)    
        
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate,
                                    weight_decay=0.03, betas=(0.99, 0.99))
        
    elif args.model == 'FLEX':
        encoder, superres_encoder, _, decoder = FLEX(
            image_size=args.patch_size, 
            in_channels=1, 
            out_channels=1)
        
        model = DiffusionModel(
            encoder=encoder.cuda(),
            decoder=decoder.cuda(),
            superres_encoder=superres_encoder.cuda(),
            n_T=args.time_steps,
            prediction_type=args.prediction_type,
            criterion=torch.nn.L1Loss()
        )

        # choose optimizer
        ema = ExponentialMovingAverage(model.parameters(), decay=0.999)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    return train_set, model, optimizer, ema


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset),
        num_workers=8,
        drop_last=True
    )


def main(rank: int, world_size: int, sampling_freq: int, epochs: int, batch_size: int, run, args):
    
    # print(rank)
    # print(world_size)
    
    # wandb.login(key="5282eaefee2cb8f881265effb6251abf1703deee")
    # run = wandb.init(
    #     # Set the project where this run will be logged
    #     project="InterpDM",
    #     name=args.run_name,
    #     mode = 'disabled',
    #     # Track hyperparameters and run metadata
    #     config={
    #         "learning_rate": args.learning_rate,
    #         "epochs": args.epochs,
    #         "batch size": args.batch_size,
    #         "upsampling factor": args.factor,
    #     },
    # )
    
    # ddp_setup(rank, world_size)
    local_rank, rank = ddp_setup(rank, world_size)
    print("local rank, rank: ", local_rank, rank)
    
    device = torch.cuda.current_device()
    print("device: ", device)
    
    dataset, model, optimizer, ema = load_train_objs(args=args)
    if ema is not None:
        model.ema = ema
    # torch.cuda.set_device(device)
    model = model.to(device)

    if args.checkpoint_path != '':
        model, ema, optimizer = load_checkpoint(args.checkpoint_path, model, optimizer, device)
        # checkpoint = torch.load(args.checkpoint_path, weights_only=True)
        # model.load_state_dict(checkpoint["model"])
        # ema = ExponentialMovingAverage(model.parameters(), decay=0.999)
        # ema.load_state_dict(checkpoint["ema"])
    
    #==============================================================================
    # Model summary
    #==============================================================================
    print('**** Setup ****')
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    print('************')

    train_data = prepare_dataloader(dataset, batch_size)
    
    # gpu_id: int, local_gpu_id: int,
    start = time.time()
    trainer = Trainer(
        model, 
        train_data, 
        optimizer, 
        gpu_id = rank, 
        local_gpu_id = local_rank, 
        sampling_freq = sampling_freq, 
        run = run, 
        run_name = args.run_name,
        use_last_snapshot = args.use_last_snapshot,
        num_pred_steps = args.num_pred_steps,
        if_normalize=args.if_normalize)
    
    trainer.train(epochs)
    end = time.time()
    print("Training time: ", end - start)
    destroy_process_group()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Minimalistic Diffusion Model for Super-resolution')
    parser.add_argument("--run_name", type=str, default='run1', help="Name of the current run.")
    parser.add_argument('--sampling_freq', default=25, type=int, help='How often to save a snapshot')
    
    # Training parameters
    parser.add_argument('--epochs', default=500, type=int, help='Total epochs to train the model')
    parser.add_argument('--batch_size', default=16, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument('--learning_rate', default=2e-4, type=float, help='learning rate')
    parser.add_argument('--checkpoint_path', default='', type=str, help='for reloading checkpoint and keep training')

    # Interpolation parameters
    parser.add_argument('--num_pred_steps', default=1, type=int, help='different prediction steps to condition on')
    parser.add_argument('--factor', default=8, type=int, help='upsampling factor')
    parser.add_argument('--patch_size', default=256, type=int, help='Patch size for the datasets') # 256
    parser.add_argument('--stride', default=128, type=int, help='Stride for the datasets') # 128
    parser.add_argument('--scratch_dir', default='/global/cfs/cdirs/m4633/foundationmodel/nskt_tensor/', type=str, help='Directory for the dataset')
    parser.add_argument('--use_last_snapshot', default=True, type=lambda x: (str(x).lower() == 'true'), help='load the last snapshot of the model')
    parser.add_argument('--if_normalize', default=True, type=lambda x: (str(x).lower() == 'true'), help='whether to normalize the data or not')

    
    # Diffusion parameters
    parser.add_argument("--prediction_type", type=str, default='v', help="Quantity to predict during training.")
    parser.add_argument("--sampler", type=str, default='ddim', help="Sampler to use to generate images")    
    parser.add_argument("--time_steps", type=int, default=10, help="Time steps for sampling")    

    # Model
    parser.add_argument("--model", type=str, default='UNetVIT', help="model")    

    # U-Net parameters
    parser.add_argument("--base_width", type=int, default=64, help="Basewidth of U-Net")    
    args = parser.parse_args()

    #parser.add_argument("--multi_node", action='store_true', default=False, help='Use multi node training')
    #parser.add_argument("--fine_tune", action='store_true', default=False, help='Fine tune using pretrained model')


    # Launch processes.
    print('Launching processes...')
    
    # wandb.login()
    wandb.login(key="5282eaefee2cb8f881265effb6251abf1703deee")
    
    run = wandb.init(
        # Set the project where this run will be logged
        # Interpolation-Diff
        project="InterpDM",
        name=args.run_name,
        # Track hyperparameters and run metadata
        config={
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "batch size": args.batch_size,
            "upsampling factor": args.factor,
            "num_pred_steps": args.num_pred_steps
        },
    )
    
    # if args.multi_node:
    #     def is_master_node():
    #         return int(os.environ['RANK']) == 0
        
    #     if is_master_node():
    #         #mode = "disabled"
    #         mode = None
    #         # wandb.login()
    #         wandb.login(key="5282eaefee2cb8f881265effb6251abf1703deee")
  
    #     else:
    #         mode = "disabled"

    #     run = wandb.init(
    #         # Set the project where this run will be logged
    #         project="InterpDM",
    #         name=args.run_name,
    #         mode = mode,
    #         # Track hyperparameters and run metadata            
    #         config={
    #             "learning_rate": args.learning_rate,
    #             "epochs": args.epochs,
    #             "batch size": args.batch_size,
    #             "upsampling factor": args.factor,
    #         },
    #     )

    #     main(0,1, args.epochs, args.batch_size, run, args)
        
    # else:
        # wandb.login()
    
    
    world_size = torch.cuda.device_count()
    print("world size: ", world_size)
    
    if world_size == 1:
        main(0, world_size, args.sampling_freq, args.epochs, args.batch_size, run, args)
    else:
        mp.spawn(main, args=(world_size, args.sampling_freq, args.epochs, args.batch_size, run, args), nprocs=world_size)


    # linear scheduler, patch size 2, no attention drop

    #export CUDA_VISIBLE_DEVICES=6,7;  python train.py --model UNetVIT --batch-size 18 --run-name UNetVIT --learning-rate 0.0002 --epochs 300

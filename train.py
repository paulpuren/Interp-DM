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
from src.lion import Lion
from diffusers.optimization import get_linear_schedule_with_warmup as scheduler
from src.unet import UNet
from src.flex import FLEX
from src.diffusion_model import DiffusionModel
# from datasets.get_data import NSKT as NSKT
from datasets.data_nskt import NSKT
from datasets.data_shanghai import Shanghai
from datasets.data_sea_temp import InputHandle
from src.plotting import plot_samples
from utils.params import get_args

import warnings
warnings.filterwarnings("ignore")

# import wandb.errors
# print(dir(wandb.errors))

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
            world_size = world_size
        )
        rank = local_rank
        
    else:
        init_process_group(
            backend="gloo", 
            init_method='env://'
        )
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
            run_name: str
        ) -> None:
        
        self.gpu_id = gpu_id
        self.local_gpu_id = local_gpu_id
        self.model = model.to(local_gpu_id)
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.sampling_freq = sampling_freq
        self.model = DDP(
            model, 
            device_ids = [local_gpu_id], 
            find_unused_parameters = True
        )
        self.run = run
        self.run_name = run_name
    
    def _run_batch(
            self, 
            targets, 
            condition_start, 
            condition_end,  
            reynolds_number,
            target_interp_step,
            total_interp_steps
        ):
        self.optimizer.zero_grad()
        # if isinstance(self.model.module, DiffusionModel):
        #     reynolds_number = reynolds_number.unsqueeze(-1)
        
        loss = self.model(
            targets, 
            condition_start, 
            condition_end, 
            reynolds_number, 
            target_interp_step,
            total_interp_steps
        )
        loss.backward()
        if isinstance(self.model.module, DiffusionModel):
            torch.nn.utils.clip_grad_norm_(self.model.module.parameters(), 1.)
        self.optimizer.step()
        self.lr_scheduler.step()
        self.model.module.ema.update()
        return loss.item()

    def _run_epoch(self, epoch):        
        self.train_loader.sampler.set_epoch(epoch)
        loss_values_task = []
        for inputs, targets, cond_params in self.train_loader:
            # Unpack the input tuple
            condition_start, condition_end = inputs
            condition_start = condition_start.to(self.local_gpu_id)
            condition_end = condition_end.to(self.local_gpu_id)
            targets = targets.to(self.local_gpu_id)
            
            # unpack the condition parameters
            target_interp_step, total_interp_steps, reynolds_number = cond_params
            reynolds_number = reynolds_number.to(self.local_gpu_id)
            target_interp_step = target_interp_step.to(self.local_gpu_id)
            total_interp_steps = total_interp_steps.to(self.local_gpu_id)
            
            loss_task = self._run_batch(
                targets, 
                condition_start, 
                condition_end, 
                reynolds_number, 
                target_interp_step,
                total_interp_steps
            )
            loss_values_task.append(loss_task)

        self.run.log({"Task loss": np.mean(loss_values_task)})
        # self.run.log({"Contrastive loss": 0})  
        return loss_values_task

    def _generate_samples(self, epoch):
        sample_dir = "./samples"
        os.makedirs(sample_dir, exist_ok = True)

        sample_path = "{}/train_samples_{}".format(
            sample_dir,
            self.run_name
        )
        os.makedirs(sample_path, exist_ok = True)

        with self.model.module.ema.average_parameters():

            self.model.eval()
            with torch.no_grad():
                self.train_loader.sampler.set_epoch(1)

                # unpack the data
                inputs, targets, cond_params = next(iter(self.train_loader))
                condition_start, condition_end = inputs
                condition_start = condition_start.to(self.local_gpu_id)
                condition_end = condition_end.to(self.local_gpu_id)
                
                # unpack the condition parameters
                target_interp_step, total_interp_steps, reynolds_number = cond_params
                reynolds_number = reynolds_number.to(self.local_gpu_id)
                target_interp_step = target_interp_step.to(self.local_gpu_id)
                total_interp_steps = total_interp_steps.to(self.local_gpu_id)
                # print(f'Type {type(target_interp_step)}')

                # if isinstance(self.model.module, DiffusionModel):
                #     reynolds_number = reynolds_number.unsqueeze(-1)
                
                samples = self.model.module.sample(
                    targets.shape[0],
                    (1, targets.shape[2], targets.shape[3]),
                    condition_start,
                    condition_end,
                    reynolds_number,
                    target_interp_step,
                    total_interp_steps,
                    self.local_gpu_id
                )
        plot_samples(samples, condition_start, targets, sample_path, epoch)
        print(f"Epoch {epoch} | Generated samples saved at {sample_path}")

    def _save_checkpoint(self, epoch, name=''):
        checkpoint_dir = "./checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)

        save_path = "{}/checkpoint_{}.pt".format(
            checkpoint_dir,
            self.run_name
        )
        save_dict = {
            'model': self.model.module.state_dict(),
            'ema': self.model.module.ema.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(save_dict, save_path)
        
        if name == '':
            print(f"Epoch {epoch} | Training checkpoint saved at {save_path}")

    def train(self, max_epochs: int):
        print('--- Starting training ---')
        self.lr_scheduler = scheduler(
            optimizer = self.optimizer,
            num_warmup_steps = len(self.train_loader) * 3, # short warmup phase
            num_training_steps = (len(self.train_loader) * max_epochs)
        )
        best_mse = np.inf
        self.model.train()
        for epoch in range(max_epochs):
            loss_values = self._run_epoch(epoch)

            if self.local_gpu_id == 0:
                avg_loss = np.mean(loss_values)
                print("Epoch {} | loss {:.4f} | learning rate {:.6f}".format(
                    epoch + 1, 
                    avg_loss, 
                    self.lr_scheduler.get_last_lr()[0]
                ))
                self.run.log({"loss": avg_loss})

                # Save the last and best checkpoint
                self._save_checkpoint(epoch + 1, name = '_last')
                
                if best_mse > avg_loss:
                    self._save_checkpoint(epoch + 1, name='_best')
                    best_mse = avg_loss

                # Generate samples at specified intervals
                if (
                    epoch == 0 or 
                    (epoch + 1) % self.sampling_freq == 0 or 
                    (epoch + 1) == max_epochs
                ):
                    self._generate_samples(epoch+1)
            
def load_checkpoint(
        save_path, 
        model, 
        optimizer, 
        device
    ):
    if not os.path.exists(save_path):
        print(f"Unable to load from {save_path}")

    checkpoint = torch.load(save_path, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    ema = ExponentialMovingAverage(model.parameters(), decay=0.999)
    ema.load_state_dict(checkpoint["ema"])
    optimizer.load_state_dict(checkpoint['optimizer'])

    print(f"Loaded model from {save_path}")
    return model, ema, optimizer
          
def load_train_objs(args):
    
    # load training set
    if args.data_name == "nskt":
        train_set = NSKT(
            patch_size = args.patch_size, 
            stride = args.stride,
            num_interp_steps= args.total_interp_steps,
            scratch_dir = args.scratch_dir,
            flag = "train",
            # train = True,
            is_T_fixed = args.is_T_fixed
        )
    elif args.data_name == "shanghai":
        train_set = Shanghai(
            data_path = args.scratch_dir,
            img_size = args.patch_size, 
            type = "train",
            trans = None,
            total_interp_steps = args.total_interp_steps
        )
    elif args.data_name == "sea_temp":
        input_param = {
            'path': args.scratch_dir,
            'total_length': args.total_interp_steps, # total length of each sample (input + output)
            'input_length': 2, # length of input sequence
            'type': 'train', # train/test/valid
            'input_data_type': 'float32'
        }
        train_set = InputHandle(input_param)
    else:
        print("This dataset is not supported.")
        sys.exit()

    ema = None # placeholder for non-FLEX model
    if args.model == 'FLEX':
        encoder, task_encoder, task_encoder_end, decoder = FLEX(
            image_size = args.patch_size, 
            in_channels = 1, 
            out_channels = 1,
            model_size= "small", # was "small", "medium"
            mlp_ratio = 2 # or maybe 4
        )
        model = DiffusionModel(
            encoder = encoder.cuda(),
            decoder = decoder.cuda(),
            task_encoder = task_encoder.cuda(),
            task_encoder_end = task_encoder_end.cuda(),
            diff_steps = args.time_steps,
            prediction_type = args.prediction_type,
            criterion = torch.nn.L1Loss() # maybe l2?
        )
        # choose optimizer
        ema = ExponentialMovingAverage(
            model.parameters(), 
            decay = 0.999
        )
    elif args.model == 'UNet':
        model = UNet(
            image_size = args.patch_size, 
            in_channels = 2, # start and end frames
            out_channels = 1, # predict interpolated frame
            base_width = args.base_width
        )
    else:
        print("This model is not supported.")
        sys.exit()

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr = args.learning_rate
        )
    elif args.optimizer == 'lion':
        optimizer = Lion(
            model.parameters(), 
            lr = args.learning_rate
        )
    else:
        print("Only Adam and Lion are supported.")
        sys.exit()
    return train_set, model, optimizer, ema

def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size = batch_size,
        pin_memory = True,
        sampler = DistributedSampler(dataset),
        shuffle = False,
        num_workers = 8,
        drop_last = True
    )


def main(
        rank: int, 
        world_size: int, 
        sampling_freq: int, 
        epochs: int, 
        batch_size: int, 
        run, 
        args
    ):
    
    local_rank, rank = ddp_setup(rank, world_size)
    print("local rank, rank: ", local_rank, rank)
    
    device = torch.cuda.current_device()
    print("device: ", device)
    
    dataset, model, optimizer, ema = load_train_objs(args = args)
    train_data = prepare_dataloader(dataset, batch_size)

    if ema is not None:
        model.ema = ema
    # torch.cuda.set_device(device)
    model = model.to(device)

    # post-training checkpoint loading
    if args.checkpoint_path != '':
        model, ema, optimizer = load_checkpoint(
            args.checkpoint_path, 
            model, 
            optimizer, 
            device
        )
    
    # Model summary
    print("**************")
    print("Total model params: %.2fM" % (
            sum(p.numel() for p in model.parameters()) / 1000000.0
        )
    )
    print("**************")
    
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
        run_name = args.run_name
    )
    trainer.train(epochs)
    end = time.time()
    print("Training time: ", end - start)
    destroy_process_group()

if __name__ == "__main__":
    args = get_args()

    # Launch processes.
    print('Launching processes...')
    
    # wandb.login()
    wandb.login(key = "5282eaefee2cb8f881265effb6251abf1703deee")
    args.run_name = "Model2s_interp_skip0.1_{}_Data_{}_Optim_{}_lr{}_epoch{}_stride{}_T{}_Tfixed{}".format(
            args.model,
            args.data_name,
            args.optimizer,
            args.learning_rate,
            args.epochs,
            args.stride,
            args.total_interp_steps,
            args.is_T_fixed
    )
    run = wandb.init(
        # Set the project where this run will be logged
        project = "InterpDM",
        name = args.run_name,
        # Track hyperparameters and run metadata
        config = {
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "batch size": args.batch_size,
            "total_interp_steps": args.total_interp_steps
        },
    )

    world_size = torch.cuda.device_count()
    print("world size: ", world_size)
    
    if world_size == 1:
        main(
            0, 
            world_size, 
            args.sampling_freq, 
            args.epochs, 
            args.batch_size, 
            run, 
            args
        )
    else:
        mp.spawn(
            main, 
            args = (
                world_size, 
                args.sampling_freq, 
                args.epochs, 
                args.batch_size, 
                run, 
                args
            ), 
            nprocs = world_size
        )


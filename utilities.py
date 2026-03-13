import os, sys
import torch
from torch_ema import ExponentialMovingAverage
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from src.lion import Lion
from src.unet import UNet
from src.flex import FLEX
from src.diffusion_model import DiffusionModel
from datasets.data_nskt import NSKT, NSKT_eval
from datasets.data_shanghai import Shanghai
from datasets.data_sea_temp import InputHandle, InputHandleEval
from pathlib import Path
from datetime import datetime

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
            num_interp_steps= args.total_interp_steps_train,
            scratch_dir = args.scratch_dir,
            flag = "train",
            is_T_fixed = args.is_T_fixed
        )
        val_set = NSKT(
            patch_size = args.patch_size, 
            stride = args.stride,
            num_interp_steps= args.total_interp_steps_train,
            scratch_dir = args.scratch_dir,
            flag = "val",
            is_T_fixed = args.is_T_fixed
        )
    #
    elif args.data_name == "shanghai":
        # ['train', 'test', 'val']
        train_set = Shanghai(
            data_path = args.scratch_dir,
            img_size = args.patch_size, 
            type = "train",
            trans = None,
            total_interp_steps = args.total_interp_steps_train
        )
        val_set = Shanghai(
            data_path = args.scratch_dir,
            img_size = args.patch_size, 
            type = "val",
            trans = None,
            total_interp_steps = args.total_interp_steps_train
        )
    #
    elif args.data_name == "sea_temp":
        train_input_param = {
            'path': args.scratch_dir,
            'total_length': args.total_interp_steps_train, # total length of each sample (input + output)
            'input_length': 2, # length of input sequence
            'type': 'train', # train/test/valid
            'input_data_type': 'float32'
        }
        val_input_param = {
            'path': args.scratch_dir,
            'total_length': args.total_interp_steps_train, # total length of each sample (input + output)
            'input_length': 2, # length of input sequence
            'type': 'valid', # train/test/valid
            'input_data_type': 'float32'
        }
        train_set = InputHandle(train_input_param)
        val_set = InputHandle(val_input_param)
    # 
    else:
        print(
            "This dataset is not supported. We currently only support (nskt), (shanghai), and (sea_temp) datasets."
        )
        sys.exit()

    ema = None # placeholder for non-diffusion-based models
    if args.model == 'FLEX':
        encoder, task_encoder, task_encoder_end, decoder = FLEX(
            image_size = args.patch_size, 
            in_channels = 1, 
            out_channels = 1,
            model_size= args.flex_model_size, # was "small", "medium"
            mlp_ratio = args.flex_mlp_ratio # or maybe 4
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
    return train_set, val_set, model, optimizer, ema


def load_eval_obj(args):
    # load evaluation set
    if args.data_name == "nskt":
        eval_set = NSKT_eval(
            patch_size=args.patch_size,
            stride=args.stride,
            num_interp_steps=args.total_interp_steps,
            re_id=args.re_id,
            scratch_dir=args.scratch_dir,
        )
    elif args.data_name == "shanghai":
        eval_set = Shanghai(
            data_path=args.scratch_dir,
            img_size=args.patch_size,
            type="test",
            trans=None,
            total_interp_steps=args.total_interp_steps,
        )
    elif args.data_name == "sea_temp":
        eval_input_param = {
            "path": args.scratch_dir,
            "total_length": args.total_interp_steps,
            "input_length": 2,
            "type": "test",
            "input_data_type": "float32",
        }
        eval_set = InputHandleEval(eval_input_param)
    else:
        print(
            "This dataset is not supported. We currently only support (nskt), (shanghai), and (sea_temp/sst) datasets."
        )
        sys.exit()

    ema = None  # placeholder for non-diffusion-based models
    if args.model == "FLEX":
        encoder, task_encoder, task_encoder_end, decoder = FLEX(
            image_size=args.patch_size,
            in_channels=1,
            out_channels=1,
            model_size=args.flex_model_size,
            mlp_ratio=args.flex_mlp_ratio,
        )
        model = DiffusionModel(
            encoder=encoder.cuda(),
            decoder=decoder.cuda(),
            task_encoder=task_encoder.cuda(),
            task_encoder_end=task_encoder_end.cuda(),
            diff_steps=args.time_steps,
            prediction_type=args.prediction_type,
            criterion=torch.nn.L1Loss(),
        )
        ema = ExponentialMovingAverage(
            model.parameters(),
            decay=0.999,
        )
    elif args.model == "UNet":
        model = UNet(
            image_size=args.patch_size,
            in_channels=2,
            out_channels=1,
            base_width=args.base_width,
        )
    else:
        print("This model is not supported.")
        sys.exit()

    return eval_set, model, ema

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

def get_run_name(args):
    if args.model == "FLEX":
        if args.checkpoint_path == '':
            run_name = "Model_{}_{}_mlp{}_Data_{}_Optim_{}_cosine_lr{}_epoch{}_stride{}_T{}_Tfixed{}".format(
                    args.model,
                    args.flex_model_size,
                    args.flex_mlp_ratio,
                    args.data_name,
                    args.optimizer,
                    args.learning_rate,
                    args.epochs,
                    args.stride,
                    args.total_interp_steps_train,
                    args.is_T_fixed
            )
        else:
            run_name = "Model_ft_{}_{}_mlp{}_Data_{}_Optim_{}_cosine_lr{}_epoch{}_stride{}_T{}_Tfixed{}".format(
                    args.model,
                    args.flex_model_size,
                    args.flex_mlp_ratio,
                    args.data_name,
                    args.optimizer,
                    args.learning_rate,
                    args.epochs,
                    args.stride,
                    args.total_interp_steps_train,
                    args.is_T_fixed
            )
    else:
        if args.checkpoint_path == '':
            run_name = "Model_{}_Data_{}_Optim_{}_lr{}_epoch{}_stride{}_T{}_Tfixed{}".format(
                    args.model,
                    args.data_name,
                    args.optimizer,
                    args.learning_rate,
                    args.epochs,
                    args.stride,
                    args.total_interp_steps_train,
                    args.is_T_fixed
            ) 
        else:
            run_name = "Model_ft_{}_Data_{}_Optim_{}_lr{}_epoch{}_stride{}_T{}_Tfixed{}".format(
                    args.model,
                    args.data_name,
                    args.optimizer,
                    args.learning_rate,
                    args.epochs,
                    args.stride,
                    args.total_interp_steps_train,
                    args.is_T_fixed
            )
    return run_name

def save_metrics(
        metrics: dict, 
        save_path: str, 
        header: str = ""
    ):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with save_path.open("a", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write(f"time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        if header:
            f.write(f"{header}\n")
        for k, v in metrics.items():
            if isinstance(v, float):
                f.write(f"{k}: {v:.6f}\n")
            else:
                f.write(f"{k}: {v}\n")

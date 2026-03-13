"""
Evaluation for NSKT fluid data
------------------------------
* eval for different Reynolds numbers
* eval for different interpolation steps
* eval for different patch sizes
* eval for different models
"""

import os, time
import torch
import numpy as np
from src.flex import FLEX
from src.diffusion_model import DiffusionModel
from src.helper import *
from torch.utils.data import Dataset, DataLoader
from torch_ema import ExponentialMovingAverage
import scipy.stats
from datasets.data_nskt import NSKT_eval
from utils.params_eval import get_args
from utilities import *

RE_EVAL_LIST = [
    600, 1000, 2000, 4000, 8000, 
    12000, 16000, 24000, 32000, 36000
]

if __name__ == "__main__":
    args = get_args()
    
    # load model
    print("Loading the trained model...")

    # FLEX model
    _, _, model, _, ema = load_train_objs(args = args)

    # model save path
    checkpoint_dir = './checkpoints'
    run_name = get_run_name(args)
    save_path = "{}/checkpoint_{}.pt".format(
        checkpoint_dir,
        run_name
    )
    print("Loading model from: ", save_path)
    checkpoint = torch.load(save_path, weights_only = True)
    model.load_state_dict(checkpoint["model"])
    if ema is not None:
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
    test_set = NSKT_eval(
        patch_size = args.patch_size,
        stride = args.stride,
        num_interp_steps = args.total_interp_steps,
        re_id = args.re_id,
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

    if ema is not None:
        with ema.average_parameters():
            with torch.no_grad():
                model.eval()
                inference_time = []
                for i, (inputs, targets, cond_params) in enumerate(testloader):
                    print(i)
                    # Unpack the input tuple
                    # [b,c,h,w] = [32, 1, 128, 128]
                    condition_start, condition_end = inputs
                    condition_start = condition_start.to('cuda')
                    condition_end = condition_end.to('cuda')
                    # print("condition_end shape: ", condition_end.shape) 
                    # print("lengths of targets: ", len(targets)) # 20
                    # print("target shape: ", targets[0].shape) # [32,1,128,128]
                    
                    # unpack the condition parameters
                    total_interp_steps, reynolds_number = cond_params
                    reynolds_number = reynolds_number.to('cuda') 
                    total_interp_steps = total_interp_steps.to('cuda')

                    preds = []
                    len_targets = len(targets)

                    start = time.time()
                    for ii in range(len(targets)): # total interp step (e.g., 20)
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
                        ) # shape: [b, c, h, w] = [32, 1, 128, 128]    
                        # print("predictions shape: ", predictions.shape)             
                        preds.append(predictions.cpu().detach().numpy())
                    end = time.time()
                    inference_time.append((end - start) / len(targets)) # each snapshot

                    # iterate over batch size: 32
                    for j in range(predictions.shape[0]):
                        RFNE_error_at_time_p = []
                        cc_error_at_time_p = []
                        
                        # total interp steps
                        for p in range(len(targets)): 

                            # data shape: [b,c,h,w]
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
                            RE_EVAL_LIST[args.re_id],
                            args.total_interp_steps
                        )
                        np.save(sample_path + '.npy', samples)
                        print('Generated samples saved...')
    else:
        with torch.no_grad():
            model.eval()
            inference_time = []
            for i, (inputs, targets, cond_params) in enumerate(testloader):
                print(i)
                # Unpack the input tuple
                # [b,c,h,w] = [32, 1, 128, 128]
                condition_start, condition_end = inputs
                condition_start = condition_start.to('cuda')
                condition_end = condition_end.to('cuda')
    
                # unpack the condition parameters
                total_interp_steps, reynolds_number = cond_params
                reynolds_number = reynolds_number.to('cuda') 
                total_interp_steps = total_interp_steps.to('cuda')

                preds = []
                len_targets = len(targets)

                start = time.time()
                for ii in range(len(targets)): # total interp step (e.g., 20)
                    target_interp_step = torch.tensor(
                        (ii + 1), 
                        dtype = torch.float32
                    ).to('cuda')
                    target_interp_step = target_interp_step.repeat(
                        condition_start.shape[0]
                    )
                    # shape: [b, c, h, w] = [32, 1, 128, 128]    
                    predictions = model.sample(
                        condition_start,
                        condition_end,
                        reynolds_number,
                        target_interp_step,
                        total_interp_steps
                    )
                    preds.append(predictions.cpu().detach().numpy())
                end = time.time()
                inference_time.append((end - start) / len(targets)) # each snapshot

                # iterate over batch size: 32
                for j in range(predictions.shape[0]):
                    RFNE_error_at_time_p = []
                    cc_error_at_time_p = []
                    
                    # total interp steps
                    for p in range(len(targets)): 

                        # data shape: [b,c,h,w]
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
                        RE_EVAL_LIST[args.re_id],
                        args.total_interp_steps
                    )
                    np.save(sample_path + '.npy', samples)
                    print('Generated samples saved...')

    avg_RFNE = np.mean(np.vstack(RFNE_error), axis=0)
    print(f'Average RFNE={repr(avg_RFNE)}')

    avg_R2 = np.mean(np.vstack(R2s), axis=0)
    print(f'Average Pearson correlation coefficients={repr(avg_R2)}')

    # print("inference time shape: ", len(inference_time))
    # print("inference time samples: ", inference_time[0].shape)
    avg_infer_time = np.mean(inference_time, axis=0)
    print(f'Average Inference Time={repr(avg_infer_time)}')

    # save results
    metrics = {
        "avg_rfne_steps": avg_RFNE, 
        "avg_r2_steps": avg_R2,
        "avg_rfne_value": np.mean(avg_RFNE, axis=0), 
        "avg_r2_value": np.mean(avg_R2, axis=0),
        "avg_run_time": avg_infer_time
    }
    metrics_save_path = "./assets/eval_re{}_{}.txt".format(
        RE_EVAL_LIST[args.re_id],
        run_name
    )
    header = "{}_RE{}".format(run_name, RE_EVAL_LIST[args.re_id])
    save_metrics(
        metrics = metrics, 
        save_path = metrics_save_path, 
        header = header # "model=resnet50, split=test"
    )



# export CUDA_VISIBLE_DEVICES=7; python evaluation.py --task forecast --batch-size 32 --horizen 50 --Reynolds-number 12000

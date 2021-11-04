# mbt2018_mean
# vimeo

#!/usr/bin/env python
# coding: utf-8

# # 1. Main training and validation code

# In[25]:


import torch
import numpy as np
import os
import sys
import warnings

warnings.filterwarnings('ignore')
import imageio
from compressai.zoo import mbt2018_mean
import wandb
import argparse
import logging
import time
import random
from torch import optim

from torch.utils.data import DataLoader, RandomSampler

model_path = os.path.abspath('..')
sys.path.insert(1, model_path)

from model import m
from utils import float_to_uint8, MSE, PSNR, calculate_distortion_loss
from utils import TrainDataset, TestDataset
from utils import image_compress, save_model
from utils import load_model, build_info, update_train_info
from utils import update_val_info, update_best_val_info, zero_train_info

torch.backends.cudnn.benchmark = True

# ### Training Function

# In[ ]:


def train_one_step(im_batch, model, optimizer, alpha, device):
    beta = 1

    model.train()
    
    frame_list = [im_batch[:, 3*i:3*(i+1)] for i in range(im_batch.shape[1] // 3)]
    x1 = frame_list[0]
    
    avg_dist_loss = torch.zeros((1)).to(device)
    avg_rate_loss = torch.zeros((1)).to(device)
    
    frame_prev = x1
    flow_lat_prev = torch.zeros((x1.shape[0], 128, x1.shape[2]//16, x1.shape[3]//16)).to(device=device)
    res_lat_prev = torch.zeros((x1.shape[0], 128, x1.shape[2]//16, x1.shape[3]//16)).to(device=device)
    flow_state_enc = torch.zeros((x1.shape[0], 2, 128, x1.shape[2]//4, x1.shape[3]//4)).to(device=device)
    flow_state_dec = torch.zeros((x1.shape[0], 2, 128, x1.shape[2]//4, x1.shape[3]//4)).to(device=device)
    res_state_enc = torch.zeros((x1.shape[0], 2, 128, x1.shape[2]//4, x1.shape[3]//4)).to(device=device)
    res_state_dec = torch.zeros((x1.shape[0], 2, 128, x1.shape[2]//4, x1.shape[3]//4)).to(device=device)
    flow_rpm_state = torch.zeros((x1.shape[0], 2, 128, x1.shape[2]//16, x1.shape[3]//16)).to(device=device)
    res_rpm_state = torch.zeros((x1.shape[0], 2, 128, x1.shape[2]//16, x1.shape[3]//16)).to(device=device)

    for i in range(1, len(frame_list)):
        
        dec_cur, flow_lat_prev, res_lat_prev, \
        flow_state_enc, flow_state_dec, \
        res_state_enc, res_state_dec, \
        flow_rpm_state, res_rpm_state, rate = model(frame_prev, frame_list[i], flow_lat_prev,
                                                    res_lat_prev, flow_state_enc,
                                                    flow_state_dec, res_state_enc, res_state_dec,
                                                    flow_rpm_state, res_rpm_state, i, True)

        dist_loss = calculate_distortion_loss(dec_cur, frame_list[i])
        loss = alpha * dist_loss + beta * rate
        
        optimizer.zero_grad()
        
        loss.backward(retain_graph=False)
        # inputs = (frame_prev, flow_lat_prev, res_lat_prev, flow_state_enc, flow_state_dec, res_state_enc, res_state_dec, flow_rpm_state, res_rpm_state)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        avg_dist_loss += dist_loss
        avg_rate_loss += rate
        
        frame_prev = dec_cur.detach()
        flow_lat_prev = flow_lat_prev.detach()
        res_lat_prev = res_lat_prev.detach()
        flow_state_enc = flow_state_enc.detach()
        flow_state_dec = flow_state_dec.detach()
        res_state_enc = res_state_enc.detach()
        res_state_dec = res_state_dec.detach()
        flow_rpm_state = flow_rpm_state.detach()
        res_rpm_state = res_rpm_state.detach()
        
        
    avg_dist_loss /= (len(frame_list) - 1)
    avg_rate_loss /= (len(frame_list) - 1)
    loss = alpha * avg_dist_loss + beta * avg_rate_loss
                
    return avg_dist_loss, avg_rate_loss, loss


# ### Validation Function

# In[ ]:


def validate(model, test_loader, image_compressor, alpha, device):
    with torch.no_grad():
        beta = 1.

        model.eval()

        average_loss = torch.zeros((1)).to(device)
        average_bpp = torch.zeros((1)).to(device)
        average_mse = torch.zeros((1)).to(device)
            
        total_pixels = 0
        total_frames = 0

        # folder_names = ["beauty", "bosphorus", "honeybee", "jockey", "ready", "shake", "yatch"]
        # folder_number = 0
            
        for video in test_loader:
            # d = 1
            name = video["name"]
            video_images = video["frames"].to(device).float()

            # os.makedirs("uvg/" + folder_names[folder_number], exist_ok=True)
            video_total_images = video_images.shape[0]

            x1 = video_images[0]
            b, c, h, w = x1.shape

            dec, _ = image_compress(x1, image_compressor)

            flow_lat_prev = torch.zeros((x1.shape[0], 128, x1.shape[2]//16, x1.shape[3]//16)).to(device=device)
            res_lat_prev = torch.zeros((x1.shape[0], 128, x1.shape[2]//16, x1.shape[3]//16)).to(device=device)
            flow_state_enc = torch.zeros((x1.shape[0], 2, 128, x1.shape[2]//4, x1.shape[3]//4)).to(device=device)
            flow_state_dec = torch.zeros((x1.shape[0], 2, 128, x1.shape[2]//4, x1.shape[3]//4)).to(device=device)
            res_state_enc = torch.zeros((x1.shape[0], 2, 128, x1.shape[2]//4, x1.shape[3]//4)).to(device=device)
            res_state_dec = torch.zeros((x1.shape[0], 2, 128, x1.shape[2]//4, x1.shape[3]//4)).to(device=device)
            flow_rpm_state = torch.zeros((x1.shape[0], 2, 128, x1.shape[2]//16, x1.shape[3]//16)).to(device=device)
            res_rpm_state = torch.zeros((x1.shape[0], 2, 128, x1.shape[2]//16, x1.shape[3]//16)).to(device=device)

            for i in range(1, video_total_images):
                x = video_images[i]

                dec, flow_lat_prev, res_lat_prev, \
                flow_state_enc, flow_state_dec, \
                res_state_enc, res_state_dec, \
                flow_rpm_state, res_rpm_state, rate, size = model(dec, x, flow_lat_prev,
                                                                  res_lat_prev, flow_state_enc,
                                                                  flow_state_dec, res_state_enc,
                                                                  res_state_dec, flow_rpm_state,
                                                                  res_rpm_state, i, False)
                dist = calculate_distortion_loss(dec, x)
                loss = alpha * dist + beta * rate
                average_loss += loss
                average_bpp += size

                uint8_real = float_to_uint8(x[0, :, :h, :w].cpu().numpy())

                uint8_dec_out = float_to_uint8(dec[0, :, :h, :w].cpu().detach().numpy())
                average_mse += MSE(uint8_dec_out.astype(np.float64), uint8_real.astype(np.float64))

                total_pixels += uint8_real.shape[0] * uint8_real.shape[1]
                # d += 1
                total_frames += 1

            # folder_number += 1

        average_mse /= total_frames
        average_psnr = PSNR(average_mse.cpu().detach().numpy(), data_range=255)
        average_loss /= total_frames
        average_bpp /= total_pixels

    return average_loss, average_psnr, average_bpp


# ### Main Function

# In[ ]:
# We just train the b-coding model
# P-frame coding model is freezed after a complete training

# Argument parser
parser = argparse.ArgumentParser()

# Hyperparameters, paths and settings are given
# prior the training and validation
parser.add_argument("--train_path", type=str, default="/datasets/vimeo_septuplet/sequences/")   # Dataset paths
parser.add_argument("--val_path", type=str, default="/userfiles/ecetin17/full_test/")
parser.add_argument("--total_train_step", type=int, default=2000000)                            # # of total iterations
parser.add_argument("--train_step", type=int, default=5000)                                     # # of iterations for recording
parser.add_argument("--learning_rate", type=float, default=1.e-4)                               # learning rate
parser.add_argument("--aux_learning_rate", type=float, default=1.e-3)
parser.add_argument("--min_lr", type=float, default=5.e-6)                                      # min. learning rate
parser.add_argument("--batch_size", type=int, default=4)                                        # Batch size
parser.add_argument("--patch_size", type=int, default=256)                                      # Train patch sizes
parser.add_argument("--train_gop_size", type=int, default=5)                                    # Train gop sizes
parser.add_argument("--val_gop_size", type=int, default=5)                                      # Validation gop sizes
parser.add_argument("--train_skip_frames", type=int, default=1)                                 # how many frames to skip in train time
parser.add_argument("--val_skip_frames", type=int, default=1)                                   # how many frames to skip in val time
parser.add_argument("--device", type=str, default="cuda")                                       # device "cuda" or "cpu"
parser.add_argument("--workers", type=int, default=4)                                           # number of workers

parser.add_argument("--alpha", type=int, default=1626)                                          # alpha for rate-distortion trade-off
parser.add_argument("--compressor_q", type=int, default=7)                                      # I-frame compressor quality factor

parser.add_argument("--pretrained", type=str, default="../checkpoint.pth")                      # save model to folder
parser.add_argument("--cont_train", type=bool, default=False)                                   # load optimizer
parser.add_argument("--wandb", type=bool, default=True)                                         # Store results in wandb
parser.add_argument("--log_results", type=bool, default=False)                                  # Store results in log

args = parser.parse_args()

args.project_name = "RLVC"

args.model_name = "RLVC_union"

logging.basicConfig(filename= args.model_name + "_" + str(args.alpha) + ".log", level=logging.INFO)


# In[ ]:


def main(args):

    device = torch.device(args.device)
    
    if args.wandb:
        wandb.init(
            project=args.project_name, 
            name=str(args.alpha) + "_" + args.model_name, 
            config=vars(args)
        )
        
    model = m.Model().to(device).float()

    checkpoint = torch.load(args.pretrained, map_location=device)
    model = load_model(model, checkpoint)
    
    for param in model.opticFlow.parameters():
        param.requires_grad = False
    for param in model.warpnet.parameters():
        param.requires_grad = False

    print(model)
    
    image_compressor = mbt2018_mean(quality=args.compressor_q, metric="mse", 
                                    pretrained=True).to(device).float()
        
    image_compressor.eval()

    # Use list of tuples instead of dict to be able to later check the elements are unique and there is no intersection
    parameters = [(n,p) for n, p in model.named_parameters()]
    
    # Make sure we don't have an intersection of parameters
    parameters_name_set = set(n for n,p in parameters)
    assert len(parameters) == len(parameters_name_set)
    
    optimizer = optim.Adam(
        (p for (n, p) in parameters if p.requires_grad),
        lr=args.learning_rate,
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, 
                                                     patience=5, min_lr=5.e-6)
                                                     
    if args.cont_train:
        optimizer, _ = load_optimizer(
                           checkpoint=checkpoint, 
                           device=device, 
                           optimizer=optimizer, 
                           aux_optimizer=None
                       )
        scheduler.load_state_dict(checkpoint["scheduler"])

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    
    if args.wandb:
        wandb.config.update({"Num. params": params})
    
    if args.log_results:
        logging.info("Num. params: " + str(params))
    
    train_dataset = TrainDataset(args.train_path, patch_size=args.patch_size,
                                 gop_size=args.train_gop_size, skip_frames=args.train_skip_frames)
    test_dataset = TestDataset(args.val_path, gop_size=args.val_gop_size, skip_frames=args.val_skip_frames)
    train_sampler = RandomSampler(train_dataset, replacement=True)

    infographic = build_info()
    
    time_start = time.perf_counter()
    
    if args.cont_train:
        iteration = checkpoint["iter"]
    else:
        iteration = 0

    infographic = build_info()
    
    time_start = time.perf_counter()
    iteration = 0

    while iteration <= args.total_train_step:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.workers)
        
        for gop_im_batch in train_loader:

            train_loss = train_one_step(gop_im_batch.to(args.device).float(), model, 
                                        optimizer, args.alpha, device)
            iteration += 1

            if iteration % args.train_step == 0:
                update_train_info(infographic, *train_loss)
                
                test_loader = DataLoader(test_dataset, batch_size=None, shuffle=False, num_workers=args.workers)
                avg_val_loss, avg_psnr, avg_bpp = validate(model, test_loader, image_compressor, args.alpha, device)

                update_val_info(infographic, avg_val_loss, avg_psnr, avg_bpp)

                scheduler.step(avg_val_loss)
                learning_rate = optimizer.param_groups[0]["lr"]

                time_end = time.perf_counter()
                duration = time_end - time_start

                if avg_val_loss < infographic["best_val_loss"]:
                    # Save every submodule of the model separately for successive training runs
                    save_model(
                        model=model, 
                        optimizer=optimizer, 
                        aux_optimizer=None, 
                        scheduler=scheduler, 
                        num_iter=iteration, 
                        save_name="../" + args.model_name + "_" + str(args.alpha) + "_cont" + ".pth"
                    )

                    update_best_val_info(infographic)
                
                if args.wandb:
                    wandb.log(
                        {
                            "Time": duration,
                            "Learning rate": learning_rate,
                            "Distortion loss": infographic["step_train_dist_loss"] / args.train_step,
                            "Rate loss": infographic["step_train_rate_loss"] / args.train_step,
                            "Train loss": infographic["step_train_loss"] / args.train_step,
                            "Validation PSNR": infographic["avg_psnr_dec"],
                            "Validation bpp": infographic["avg_bpp"],
                            "Validation loss": infographic["avg_val_loss"],
                            "Best Validation loss": infographic["best_val_loss"],
                            "PSNR at best Validation loss": infographic["psnr_dec_at_best_loss"],
                            "bpp at best Validation loss": infographic["bpp_at_best_loss"],
                        },
                        step=iteration,
                    )
                    
                if args.log_results:
                    logging.info("Iteration: " + str(iteration))
                    logging.info("Time: " + str(duration))
                    logging.info("Learning rate: " + str(learning_rate))
                    logging.info("Distortion loss: " + str(infographic["step_train_dist_loss"] / args.train_step))
                    logging.info("Rate loss: " + str(infographic["step_train_rate_loss"] / args.train_step))
                    logging.info("Train loss: " + str(infographic["step_train_loss"] / args.train_step))
                    logging.info("Validation PSNR: " + str(infographic["avg_psnr_dec"]))
                    logging.info("Validation bpp: " + str(infographic["avg_val_loss"]))
                    logging.info("Validation loss: " + str(infographic["avg_val_loss"]))
                    logging.info("Best Validation loss: " + str(infographic["best_val_loss"]))
                    logging.info("PSNR at best Validation loss: " + str(infographic["psnr_dec_at_best_loss"]))
                    logging.info("bpp at best Validation loss: " + str(infographic["bpp_at_best_loss"]))
                    logging.info("*********************************")

                zero_train_info(infographic)

                time_start = time.perf_counter()
            
            if iteration >= args.total_train_step:
                break


# In[ ]:


if __name__ == '__main__':
    main(args)


# In[ ]:





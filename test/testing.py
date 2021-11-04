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
from utils import TestDataset
from utils import image_compress, save_model
from utils import load_model, build_info, update_train_info
from utils import update_val_info, update_best_val_info, zero_train_info

torch.backends.cudnn.benchmark = True

# ### Test Function

# In[ ]:


def test(model, image_compressor, device, args, bidir):
    with torch.no_grad():
        model.eval()
        
        if bidir:
            groups = 2
        else:
            groups = 1
            
        average_bpp = 0
        average_psnr = 0
            
        total_pixels = 0
        total_frames = 0

        folder_names = ["beauty", "bosphorus", "honeybee", "jockey", "ready", "shake", "yatch"]

        for video_num in range(len(folder_names)):
        
            # GOP size is already increased by 1 in main to get the backward I frame
            test_dataset = TestDataset(args.test_path, video_num, gop_size=args.test_gop_size, 
                                       skip_frames=args.test_skip_frames, bidirectional=bidir)
            
            test_loader = DataLoader(test_dataset, batch_size=None, shuffle=False, num_workers=args.workers)
            
            for video in test_loader:
                batch_first_frame = video["first_frame"]
                
                video_images = video["frames"].to(device).float()
            
                # os.makedirs("uvg/" + folder_names[folder_number], exist_ok=True)
                video_total_images = video_images.shape[0]
               
                # If forward not possible (too few images)
                if (video_total_images) < (args.test_gop_size//groups):
                    pass
                else:
                    
                    logging.info("Starting test with 1st frame: " + batch_first_frame)
                
                    x1 = video_images[0]
                    b, c, h, w = x1.shape
                    
                    dec_f, size_f = image_compress(x1, image_compressor)
                    uint8_dec_out = float_to_uint8(dec_f[0, :, :h, :w].cpu().detach().numpy())
                    uint8_real = float_to_uint8(x1[0, :, :h, :w].cpu().numpy())
                    
                    average_psnr += PSNR(MSE(uint8_dec_out.astype(np.float64), uint8_real.astype(np.float64)), data_range=255)
                    average_bpp += size_f
                    
                    total_pixels += uint8_real.shape[0] * uint8_real.shape[1]
                    total_frames += 1
                    
                    # Forward
                    flow_lat_prev_f = torch.zeros((x1.shape[0], 128, x1.shape[2]//16, x1.shape[3]//16)).to(device=device)
                    res_lat_prev_f = torch.zeros((x1.shape[0], 128, x1.shape[2]//16, x1.shape[3]//16)).to(device=device)
                    flow_state_enc_f = torch.zeros((x1.shape[0], 2, 128, x1.shape[2]//4, x1.shape[3]//4)).to(device=device)
                    flow_state_dec_f = torch.zeros((x1.shape[0], 2, 128, x1.shape[2]//4, x1.shape[3]//4)).to(device=device)
                    res_state_enc_f = torch.zeros((x1.shape[0], 2, 128, x1.shape[2]//4, x1.shape[3]//4)).to(device=device)
                    res_state_dec_f = torch.zeros((x1.shape[0], 2, 128, x1.shape[2]//4, x1.shape[3]//4)).to(device=device)
                    flow_rpm_state_f = torch.zeros((x1.shape[0], 2, 128, x1.shape[2]//16, x1.shape[3]//16)).to(device=device)
                    res_rpm_state_f = torch.zeros((x1.shape[0], 2, 128, x1.shape[2]//16, x1.shape[3]//16)).to(device=device)
                  
                    for i in range(1, video_total_images//groups):
                        x = video_images[i]
                        
                        dec_f, flow_lat_prev_f, res_lat_prev_f, \
                        flow_state_enc_f, flow_state_dec_f, \
                        res_state_enc_f, res_state_dec_f, \
                        flow_rpm_state_f, res_rpm_state_f, rate_f, size_f = model(dec_f, x, flow_lat_prev_f,
                                                                                  res_lat_prev_f, flow_state_enc_f,
                                                                                  flow_state_dec_f, res_state_enc_f, res_state_dec_f,
                                                                                  flow_rpm_state_f, res_rpm_state_f, i, False)
        
                        uint8_real = float_to_uint8(x[0, :, :h, :w].cpu().numpy())
        
                        uint8_dec_out = float_to_uint8(dec_f[0, :, :h, :w].cpu().detach().numpy())
                        average_psnr += PSNR(MSE(uint8_dec_out.astype(np.float64), uint8_real.astype(np.float64)), data_range=255)
                        average_bpp += size_f
        
                        total_pixels += uint8_real.shape[0] * uint8_real.shape[1]
                        total_frames += 1
                
                # Backward 
                # If backward coding is not possible (too few images for backward)
                if bidir:
                    if (video_images.shape[0]) != (args.test_gop_size):
                        pass
                    else:
                        x_last = video_images[-1]
                        dec_b, size_b = image_compress(x_last, image_compressor)
                        
                        uint8_dec_out = float_to_uint8(dec_b[0, :, :h, :w].cpu().detach().numpy())
                        uint8_real = float_to_uint8(x_last[0, :, :h, :w].cpu().numpy())
                        
                        average_psnr += PSNR(MSE(uint8_dec_out.astype(np.float64), uint8_real.astype(np.float64)), data_range=255)
                        average_bpp += size_b
                        
                        total_pixels += uint8_real.shape[0] * uint8_real.shape[1]
                        total_frames += 1
                                           
                        flow_lat_prev_b = torch.zeros((x1.shape[0], 128, x1.shape[2]//16, x1.shape[3]//16)).to(device=device)
                        res_lat_prev_b = torch.zeros((x1.shape[0], 128, x1.shape[2]//16, x1.shape[3]//16)).to(device=device)
                        flow_state_enc_b = torch.zeros((x1.shape[0], 2, 128, x1.shape[2]//4, x1.shape[3]//4)).to(device=device)
                        flow_state_dec_b = torch.zeros((x1.shape[0], 2, 128, x1.shape[2]//4, x1.shape[3]//4)).to(device=device)
                        res_state_enc_b = torch.zeros((x1.shape[0], 2, 128, x1.shape[2]//4, x1.shape[3]//4)).to(device=device)
                        res_state_dec_b = torch.zeros((x1.shape[0], 2, 128, x1.shape[2]//4, x1.shape[3]//4)).to(device=device)
                        flow_rpm_state_b = torch.zeros((x1.shape[0], 2, 128, x1.shape[2]//16, x1.shape[3]//16)).to(device=device)
                        res_rpm_state_b = torch.zeros((x1.shape[0], 2, 128, x1.shape[2]//16, x1.shape[3]//16)).to(device=device)
                        
                        for i in range(video_total_images-2, video_total_images//groups-1, -1):
                            x = video_images[i]
                            
                            dec_b, flow_lat_prev_b, res_lat_prev_b, \
                            flow_state_enc_b, flow_state_dec_b, \
                            res_state_enc_b, res_state_dec_b, \
                            flow_rpm_state_b, res_rpm_state_b, rate_b, size_b = model(dec_b, x, flow_lat_prev_b,
                                                                                      res_lat_prev_b, flow_state_enc_b,
                                                                                      flow_state_dec_b, res_state_enc_b, res_state_dec_b,
                                                                                      flow_rpm_state_b, res_rpm_state_b, i, False)
                            
                            uint8_real = float_to_uint8(x[0, :, :h, :w].cpu().numpy())
            
                            uint8_dec_out = float_to_uint8(dec_b[0, :, :h, :w].cpu().detach().numpy())
                            average_psnr += PSNR(MSE(uint8_dec_out.astype(np.float64), uint8_real.astype(np.float64)), data_range=255)
                            average_bpp += size_b
            
                            total_pixels += uint8_real.shape[0] * uint8_real.shape[1]
                            total_frames += 1
                        
        average_psnr /= total_frames
        average_bpp /= total_pixels

    return average_psnr.item(), average_bpp.item()


# ### Main Function

# In[ ]:
# We just train the b-coding model
# P-frame coding model is freezed after a complete training

# Argument parser
parser = argparse.ArgumentParser()

# Hyperparameters, paths and settings are given
# prior the training and validation
parser.add_argument("--test_path", type=str, default="/userfiles/ecetin17/full_test/")
parser.add_argument("--test_gop_size", type=int, default=13)                                    # test gop sizes
parser.add_argument("--test_skip_frames", type=int, default=1)                                  # how many frames to skip in test time
parser.add_argument("--device", type=str, default="cuda")                                       # device "cuda" or "cpu"
parser.add_argument("--workers", type=int, default=4)                                           # number of workers

parser.add_argument("--bidir", type=bool, default=True)                                         # bidirectional

parser.add_argument("--alpha", type=int, default=1626)                                          # alpha for rate-distortion trade-off
parser.add_argument("--compressor_q", type=int, default=7)                                      # I-frame compressor quality factor

parser.add_argument("--pretrained", type=str, default="../RLVC_union_1626_cont.pth")            # save model to folder
parser.add_argument("--wandb", type=bool, default=True)                                         # Store results in wandb
parser.add_argument("--log_results", type=bool, default=True)                                   # Store results in log

args = parser.parse_args()

args.project_name = "RLVC_test"

args.test_name = "RLVC_union_skip" + str(args.test_skip_frames) + "_gop" + str(args.test_gop_size)

logging.basicConfig(filename=args.test_name + "_" + str(args.alpha) + ".log", level=logging.INFO)

# In[ ]:


def main(args):

    device = torch.device(args.device)
    
    # Group name stays the same, change the name for different (or same) trials
    if args.wandb:
        wandb.init(project=args.project_name, name=str(args.alpha) + "_" + args.test_name, config=vars(args))

    model = m.Model().to(device).float()
    pretrained_dict = torch.load(args.pretrained)
    model = load_model(model, pretrained_dict)

    image_compressor = mbt2018_mean(quality=args.compressor_q, metric="mse", 
                                    pretrained=True).to(device).float()
    
    model.requires_grad = False
    
    image_compressor.eval()

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    
    if args.wandb:
        wandb.config.update({"Num. params": params})
        
    if args.log_results:
        logging.info("Num. params: " + str(params))
    
    time_start = time.perf_counter()

    # GOP size is increased by 1 in main to get the backward I frame
    if args.bidir:
        args.test_gop_size += 1
        
    avg_psnr, avg_bpp = test(model, image_compressor, device, args, bidir=args.bidir)

    time_end = time.perf_counter()
    duration = time_end - time_start
    
    if args.wandb:
        rd_data = [[avg_bpp, avg_psnr]]
        
        # Match column names with chart axes
        rd_table = wandb.Table(data=rd_data, columns=["bpp", "PSNR"])
        
        # "bpp vs PSNR" sets the table name, to match the tables, add them in same named table!!
        # title changes the chart title, axes are "bpp" and "PSNR"
        wandb.log({"bpp vs PSNR": wandb.plot.scatter(rd_table, "bpp", "PSNR", title="RD Curve")})
        
        time_data = [[args.test_name, duration]]
        time_table = wandb.Table(data=time_data, columns = ["Model", "Time (sec)"])
        wandb.log({"Duration": wandb.plot.bar(time_table, "Model", "Time (sec)", title="Duration of Test")})
    
    if args.log_results:
        logging.info("Average PSNR: " + str(avg_psnr))
        logging.info("Average bpp: " + str(avg_bpp))
        logging.info("Duration (sec): " + str(duration))


# In[ ]:


if __name__ == '__main__':
    main(args)


# In[ ]:





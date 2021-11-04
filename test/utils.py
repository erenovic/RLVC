#!/usr/bin/env python
# coding: utf-8

# # 2. Utils

# In[ ]:


import torch
import numpy as np
from natsort import natsorted
import glob
import random
import imageio
import math
import torch.nn as nn
import logging


# In[1]:


def normalize(tensor):
    norm = (tensor) / 255.
    return norm


# In[2]:


def float_to_uint8(image):
    clip = np.clip(image, 0, 1) * 255.
    im_uint8 = np.round(clip).astype(np.uint8).transpose(1, 2, 0)
    return im_uint8


# In[3]:


def MSE(gt, pred):
    mse = np.mean((gt - pred) ** 2)
    return mse


# In[4]:


def PSNR(mse, data_range):
    psnr = 10 * np.log10((data_range ** 2) / mse)
    return psnr


# In[5]:


def calculate_distortion_loss(out, real):
    """Mean Squared Error"""
    distortion_loss = torch.mean((out - real) ** 2)
    return distortion_loss


# In[6]:


def pad(im):
    """Padding to fix size at validation"""
    (m, c, w, h) = im.size()

    p1 = (64 - (w % 64)) % 64
    p2 = (64 - (h % 64)) % 64

    pad = nn.ReflectionPad2d(padding=(0, p2, 0, p1))
    return pad(im)


# ### Training & Test Video & Image Datasets


from torch.utils.data import Dataset


class TestDataset(Dataset):
    """Dataset for UVG"""

    def __init__(self, data_path, video_num, gop_size, skip_frames, test_size=None, bidirectional=False):
        """
        data_path: path to folders of videos,
        gop_size: how many frames to take,
        skip_frames: do we skip frames (int),
        transforms: transformation functions to be applied (array)
        """

        self.data_path = data_path
        videos = natsorted(glob.glob(data_path + "/*"))
        self.video = videos[video_num]
        video_im_list = natsorted(glob.glob(self.video + "/*.png"))
            
        # Batch size of each batch
        max_frames_in_batch = gop_size
        
        # Form batches for test
        total_num_frames = len(video_im_list)
        jump_len = gop_size*skip_frames
        self.video_splits = [video_im_list[init_frame::skip_frames][:max_frames_in_batch] 
                             for init_frame in range(0, total_num_frames-jump_len, jump_len)]
                             
        # Limiting # of GOPs in the dataset if test_size available
        if test_size:
            self.video_splits = self.video_splits[:test_size]
        
        self.gop_size = gop_size

        self.dataset_size = len(self.video_splits)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, item):
        
        video_split = self.video_splits[item]
        
        video_first_frame = video_split[0]
        
        frames = []

        for frame in video_split:
            im = imageio.imread(frame).transpose(2, 0, 1)
            (c, h, w) = im.shape
            im = im.reshape(1, c, h, w)

            im = normalize(torch.from_numpy(im))
            im = pad(im)

            frames.append(im)

        video_frames = torch.stack(frames, dim=0)
        
        return {"first_frame": video_first_frame,
                "frames": video_frames}
    
#For Youtube dataset but doesnt work because of 720x1280 shape and I-Frame compression part
#def test_image_loader(path, num_tests=7):
#    videos = natsorted(glob.glob(path + "/*"))
#    vid_list = get_batch(videos, num_tests)
#    images_list = []
#
#    for video in vid_list:
#        l = natsorted(glob.glob(video + "/*.jpg"))[::2]
#        images_list.append(l)
#    return images_list


# ### I-Frame image compressor

# In[11]:


def image_compress(im, compressor):
    out1 = compressor(im)
    dec1 = out1["x_hat"]
    size_image = sum(
        (torch.log(likelihoods).sum() / (-math.log(2)))
        for likelihoods in out1["likelihoods"].values()
    )

    return dec1, size_image


# ### Save and load pmodel

# In[12]:


def save_model(model, optimizer, aux_optimizer, num_iter, save_name="checkpoint.pth"):
    save_dict = {}
    if optimizer:
        save_dict["optimizer"] = optimizer.state_dict()
    if aux_optimizer:
        save_dict["aux_optimizer"] = aux_optimizer.state_dict()
        
    if num_iter:
        save_dict["iter"] = num_iter
    
    for child, module in model.named_children():
        save_dict[child] = module.state_dict()
        logging.info("Saved " + child + " at " + save_name)
        
    torch.save(save_dict, save_name)


# In[14]:


def load_model(model, pretrained_dict):
    model_child_names = [name for name, _ in model.named_children()]
    
    for name, submodule in pretrained_dict.items():
        if name in model_child_names:
            message = getattr(model, name).load_state_dict(submodule)
            logging.info(name + ": " + str(message))
    return model


# ### Info passing during training and validation

# In[15]:


def build_info():
    return {"step_train_dist_loss": 0,
            "step_train_rate_loss": 0,
            "step_train_loss": 0,
            "avg_psnr_dec": 0,
            "avg_bpp": 0,
            "avg_val_loss": 0,
            "best_val_loss": 10**10,
            "psnr_dec_at_best_loss": -1,
            "bpp_at_best_loss": -1
    }


# In[16]:


def update_train_info(infos, distortion_loss, rate_loss, loss):
    infos["step_train_dist_loss"] += distortion_loss.item()
    infos["step_train_rate_loss"] += rate_loss.item()
    infos["step_train_loss"] += loss.item()


# In[17]:


def zero_train_info(infos):
    infos["step_train_dist_loss"] = 0
    infos["step_train_rate_loss"] = 0
    infos["step_train_loss"] = 0


# In[18]:


def update_val_info(infos, avg_val_loss, avg_psnr, avg_bpp):
    infos["avg_val_loss"] = avg_val_loss.item()
    infos["avg_psnr_dec"] = avg_psnr.item()
    infos["avg_bpp"] = avg_bpp.item()


# In[19]:


def update_best_val_info(infos):
    infos["best_val_loss"] = infos["avg_val_loss"]
    infos["psnr_dec_at_best_loss"] = infos["avg_psnr_dec"]
    infos["bpp_at_best_loss"] = infos["avg_bpp"]


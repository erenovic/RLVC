#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Model itself

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from compressai.entropy_models import GaussianConditional

from .RAE import RAE_Encoder, RAE_Decoder
from .endecoder import ME_Spynet, flow_warp, Warp_net
from .RPM import RPM
from .functions import BitEstimator


# In[ ]:


device = torch.device("cuda")


# In[ ]:


class Model(nn.Module):
    def __init__(self, N=128, M=128, Height=256, Width=256, batch=4, act=nn.ReLU):
        super(Model, self).__init__()

        self.opticFlow = ME_Spynet()
        self.warpnet = Warp_net()

        self.flow_encoder = RAE_Encoder(in_dim=2, N=N, M=M, kernel=3, Height=Height, 
                                        Width=Width, batch=batch, act_rnn=act)
        self.flow_decoder = RAE_Decoder(out_dim=2, N=N, M=M, kernel=3, Height=Height, 
                                        Width=Width, batch=batch, act_rnn=act)
        self.residual_encoder = RAE_Encoder(in_dim=3, N=N, M=M, kernel=5, Height=Height, 
                                            Width=Width, batch=batch, act_rnn=act)
        self.residual_decoder = RAE_Decoder(out_dim=3, N=N, M=M, kernel=5, Height=Height, 
                                            Width=Width, batch=batch, act_rnn=act)

        self.residual_RPM = RPM(in_dim=128, N=N, M=M, kernel=3, stride=1, Height=Height, 
                                Width=Width, batch=batch, act_cnn=nn.ReLU, act_rnn=nn.Tanh)
        self.flow_RPM = RPM(in_dim=128, N=N, M=M, kernel=3, stride=1, Height=Height, 
                            Width=Width, batch=batch, act_cnn=nn.ReLU, act_rnn=nn.Tanh)
        
        self.ff_residual = BitEstimator(channel=128)
        self.ff_flow = BitEstimator(channel=128)
        
        self.flow_gaussian_conditional = GaussianConditional(None)
        self.res_gaussian_conditional = GaussianConditional(None)
    
    def motioncompensation(self, ref, mv):
        warpframe = flow_warp(ref, mv)
        inputfeature = torch.cat((warpframe, ref), 1)
        prediction = self.warpnet(inputfeature) + warpframe
        return prediction, warpframe
        
    def forward(self, x_previous, x_current, flow_lat_prev, res_lat_prev, rae_state_flow_encoder,
                rae_state_flow_decoder, rae_state_residual_encoder, rae_state_residual_decoder,
                rpm_state_flow, rpm_state_residual, frame_num, train):
        N, C, H, W = x_current.size()
        num_pixels = N * H * W

        # Encode & decode the flow
        mv_p2c = self.opticFlow(x_current, x_previous)

        # mv_p2c shape: (B, 2, H, W)
        # rae_state_flow_encoder shape: Initially None, later (2, B, M, H//4, W//4)

        flow_enc_result = self.flow_encoder(mv_p2c, rae_state_flow_encoder)
        flow_encoded = flow_enc_result["y"]
        # flow_encoded shape: (B, M, H//16, W//16)
  
        next_state_flow_encoder = flow_enc_result["rae_state_encoder"]        
        # next_state_flow_encoder shape: None or (2, B, M, H//4, W//4)

        # Get the entropy parameters of flow via RPM
        flow_params = self.flow_RPM(flow_lat_prev, rpm_state_flow)
        flow_mu = flow_params["mu"]
        flow_scale = flow_params["scale"]
        next_rpm_state_flow = flow_params["h_state"]

        # Estimate bits for the flow
        if frame_num == 1:
            if train:
                enc_N, enc_C, enc_H, enc_W = flow_encoded.shape
                quant_noise = torch.zeros(enc_N, enc_C, enc_H, enc_W).to(device)
                quant_noise = torch.nn.init.uniform_(torch.zeros_like(quant_noise), -0.5, 0.5)
                
                flow_encoded_hat = flow_encoded + quant_noise
            else:
                flow_encoded_hat = torch.round(flow_encoded)
                    
            size_flow, _ = self.iclr18_estimate_bits(flow_encoded_hat, flow=True)
            rate_flow = size_flow / num_pixels
        else:
            flow_encoded_hat, flow_likelihoods = self.flow_gaussian_conditional(flow_encoded,
                                                                                flow_scale, means=flow_mu)
        
        # Decode the flow
        flow_dec_result = self.flow_decoder(flow_encoded_hat, rae_state_flow_decoder)
        flow_hat = flow_dec_result["x_hat"]
        next_state_flow_decoder = flow_dec_result["rae_state_decoder"]

        # Apply motion compensation and post processing
        prediction, warpframe = self.motioncompensation(x_previous, flow_hat)
        
        residual = x_current - prediction

        # Encode residual
        res_enc_result = self.residual_encoder(residual, rae_state_residual_encoder)
        res_encoded = res_enc_result["y"]
        next_state_res_encoder = res_enc_result["rae_state_encoder"]

        # Get the entropy parameters of residual via RPM
        residual_params = self.residual_RPM(res_lat_prev, rpm_state_residual)
        residual_mu = residual_params["mu"]
        residual_scale = residual_params["scale"]
        next_rpm_state_res = residual_params["h_state"]

        # Estimate bits for the residual
        if frame_num == 1: 
            if train:
                enc_N, enc_C, enc_H, enc_W = res_encoded.shape
                quant_noise = torch.zeros(enc_N, enc_C, enc_H, enc_W).to(device)
                quant_noise = torch.nn.init.uniform_(torch.zeros_like(quant_noise), -0.5, 0.5)
                
                res_encoded_hat = res_encoded + quant_noise
            else:
                res_encoded_hat = torch.round(res_encoded)
            
            size_residual, _ = self.iclr18_estimate_bits(res_encoded_hat, flow=False)
            rate_residual = size_residual / num_pixels
        else:
            res_encoded_hat, res_likelihoods = self.res_gaussian_conditional(res_encoded,
                                                                             residual_scale, means=residual_mu)

        # Decode the residual
        residual_dec_result = self.residual_decoder(res_encoded_hat, rae_state_residual_decoder)
        residual_hat = residual_dec_result["x_hat"]
        next_state_res_decoder = residual_dec_result["rae_state_decoder"]

        x_current_hat = residual_hat + prediction
        
        if frame_num != 1:
            rate_flow = torch.log(flow_likelihoods).sum() / (-math.log(2) * num_pixels)
            size_flow = torch.log(flow_likelihoods).sum() / (-math.log(2))
            rate_residual = torch.log(res_likelihoods).sum() / (-math.log(2) * num_pixels)
            size_residual = torch.log(res_likelihoods).sum() / (-math.log(2))

        if train:
            return x_current_hat, flow_encoded_hat, res_encoded_hat, next_state_flow_encoder, \
            next_state_flow_decoder, next_state_res_encoder, next_state_res_decoder, \
            next_rpm_state_flow, next_rpm_state_res, (rate_flow + rate_residual) / 2.0
        else:
            return x_current_hat, flow_encoded_hat, res_encoded_hat, next_state_flow_encoder, \
            next_state_flow_decoder, next_state_res_encoder, next_state_res_decoder, \
            next_rpm_state_flow, next_rpm_state_res, (rate_flow + rate_residual) / 2.0, size_flow + size_residual
              
            
    def iclr18_estimate_bits(self, z, flow=True):
        """Estimate total bits, 
        https://github.com/liujiaheng/iclr_17_compression/blob/bce72bc53f5c5f5da14a54a51eedde5eb8d0c39c/model.py"""
        
        if flow:
            prob = self.ff_flow(z + 0.5) - self.ff_flow(z - 0.5)
        else:
            prob = self.ff_residual(z + 0.5) - self.ff_residual(z - 0.5)
            
        total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-10) / math.log(2.0), 0, 50))
        return total_bits, prob
        
        
    def bit_est(self, latent, mu, sigma, tiny=1.e-10):
        """Estimate total bit by Ren Yang,
        https://github.com/RenYang-home/RLVC/blob/80d32030659a758eb157e99584cc48fe2e3192b5/CNN_recurrent.py#L5"""
          
        half = torch.tensor(.5, dtype=torch.float32)
        
        upper = latent + half
        lower = latent - half
        
        sig = torch.maximum(sigma, torch.tensor(-7.0))
        upper_l = F.sigmoid((upper - mu) * (torch.exp(-sig) + tiny))
        lower_l = F.sigmoid((lower - mu) * (torch.exp(-sig) + tiny))
        p_element = upper_l - lower_l
        p_element = torch.clamp(p_element, tiny, 1-tiny)
        
        ent = - torch.log(p_element) / torch.log(torch.tensor(2.0))
        bits = torch.sum(ent)
        
        return bits, sigma, mu
        


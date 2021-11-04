# Recurrent Probability Model

import torch
import torch.nn as nn
# from compressai.models import MeanScaleHyperprior
# from compressai.layers import GDN

from .functions import conv, one_step_rnn, init_xavier


device = torch.device("cuda")


class cnn_layers(nn.Module):
    def __init__(self, in_dim=2, N=128, M=128, kernel=3, stride=1, act=nn.ReLU):
        super(cnn_layers, self).__init__()
        
        self.conv1 = conv(in_dim, N, kernel, stride)
        init_xavier(self.conv1)
        self.act1 = act()
        self.conv2 = conv(N, N, kernel, stride)
        init_xavier(self.conv2)
        self.act2 = act()
        self.conv3 = conv(N, N, kernel, stride)
        init_xavier(self.conv3)
        self.act3 = act()
        self.conv4 = conv(N, M, kernel, stride)
        init_xavier(self.conv4)
        
    def forward(self, x):
        out1 = self.act1(self.conv1(x))
        out2 = self.act2(self.conv2(out1))
        out3 = self.act3(self.conv3(out2))
        out4 = self.conv4(out3)
        return out4

# Github'ta activation olarak ReLU kullanmışlar

# Kernel Size is 5 for Residual Bottleneck, 3 for Flow Bottleneck
# Input channel number is 3 for Residual Bottleneck, 2 for Flow Bottleneck
class RPM(nn.Module):
    def __init__(self, in_dim=128, N=128, M=128, kernel=3, stride=1, Height=256, 
                 Width=256, batch=4, act_cnn=nn.ReLU, act_rnn=nn.Tanh):
        super(RPM, self).__init__()

        self.cnn_layers1 = cnn_layers(in_dim, N, M, kernel, stride, act_cnn)
        
        self.crnn = one_step_rnn(input_channels=M, num_features=M,
                                 activation=act_rnn, kernel=kernel, 
                                 batch=batch, Height=Height,
                                 Width=Width, scale=16)
        self.cnn_layers2 = cnn_layers(M, M, 2*M, kernel, stride, act_cnn)

    def forward(self, y_prev_hat, h_state):
        
        params = self.cnn_layers1(y_prev_hat)
        params, h_state = self.crnn(params, h_state)
        params = self.cnn_layers2(params)

        mu, scale = torch.chunk(params, 2, dim=1)

        return {
            "mu": mu,
            "scale": scale,
            "h_state": h_state,
        }
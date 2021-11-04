# Recurrent Autoencoder

import torch
import torch.nn as nn

# from compressai.models import MeanScaleHyperprior
from compressai.layers import GDN
from compressai.entropy_models import GaussianConditional

from .functions import conv, deconv, one_step_rnn
# from .functions import BitEstimator
from .RPM import RPM

device = torch.device("cuda")


# Kernel Size is 5 for Residual Bottleneck, 3 for Flow Bottleneck
# Input channel number is 3 for Residual Bottleneck, 2 for Flow Bottleneck
class RAE_Encoder(nn.Module):
    def __init__(self, in_dim=3, N=128, M=128, kernel=3, Height=256, 
                 Width=256, batch=4, act_rnn=nn.ReLU):
        super(RAE_Encoder, self).__init__()

        self.g_a_conv1 = conv(in_dim, N, kernel)
        self.g_a_gdn1 = GDN(N)
        self.g_a_conv2 = conv(N, N, kernel)
        self.g_a_gdn2 = GDN(N)
        self.crnn_encoder = one_step_rnn(input_channels=N, num_features=N, 
                                         activation=act_rnn, kernel=kernel, 
                                         batch=batch, Height=Height, 
                                         Width=Width, scale=4)

        self.g_a_conv3 = conv(N, N, kernel)
        self.g_a_gdn3 = GDN(N)
        self.g_a_conv4 = conv(N, M, kernel)

    def forward(self, x, rae_encoder_state):

        y = self.g_a_gdn1(self.g_a_conv1(x))
        # Reduced dim by /2
        # y shape: (B, N, H//2, W//2)

        y = self.g_a_gdn2(self.g_a_conv2(y))
        # Reduced dim by /2
        # y shape: (B, N, H//4, W//4)

        y, h_encoder = self.crnn_encoder(y, rae_encoder_state)
        # y shape: (B, N, H//4, W//4)
        # h_encoder shape: (2, B, N, H//4, W//4)
        
        y = self.g_a_gdn3(self.g_a_conv3(y))
        # Reduced dim by /2
        # y shape: (B, N, H//8, W//8)

        y = self.g_a_conv4(y)
        # Reduced dim by /2
        # y shape: (B, N, H//16, W//16)

        return {
            "y": y,
            "rae_state_encoder": h_encoder
        }


class RAE_Decoder(nn.Module):
    def __init__(self, out_dim=3, N=128, M=128, kernel=3, Height=256, 
                 Width=256, batch=4, act_rnn=nn.ReLU):
        super(RAE_Decoder, self).__init__()

        self.g_s_conv1 = deconv(M, N, kernel)
        self.g_s_gdn1 = GDN(N, inverse=True)
        self.g_s_conv2 = deconv(N, N, kernel)
        self.g_s_gdn2 = GDN(N, inverse=True)
        
        self.crnn_decoder = one_step_rnn(input_channels=N, num_features=N, 
                                         activation=act_rnn, kernel=kernel, 
                                         batch=batch, Height=Height, 
                                         Width=Width, scale=4)

        self.g_s_conv3 = deconv(N, N, kernel)
        self.g_s_gdn3 = GDN(N, inverse=True)
        self.g_s_conv4 = deconv(N, out_dim, kernel)

    def forward(self, y_hat, rae_decoder_state):

        x_hat = self.g_s_gdn1(self.g_s_conv1(y_hat))
        x_hat = self.g_s_gdn2(self.g_s_conv2(x_hat))

        x_hat, h_decoder = self.crnn_decoder(x_hat, rae_decoder_state)

        x_hat = self.g_s_gdn3(self.g_s_conv3(x_hat))
        x_hat = self.g_s_conv4(x_hat)

        return {
            "x_hat": x_hat,
            "rae_state_decoder": h_decoder,
        }
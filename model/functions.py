# Functions used in RAE, RPM and BitEstimator

import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda")


def conv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )


def deconv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        output_padding=stride - 1,
        padding=kernel_size // 2,
    )


def init_xavier(module):
    """Xavier initialize the Conv2d weights"""
    if isinstance(module, nn.Conv2d):
        nn.init.xavier_uniform_(module.weight)


class one_step_rnn(nn.Module):
    """
    Paper we are looking at:
    https://github.com/RenYang-home/RLVC/blob/master/CNN_recurrent.py
    """

    def __init__(self, input_channels, num_features, activation, 
                 kernel=3, batch=4, Height=256, Width=256, scale=4):
        super(one_step_rnn, self).__init__()

        self.cell = ConvLSTMCell(shape=(batch, Height//scale, Width//scale), 
                                 num_features=num_features, kernel=kernel, 
                                 input_channels=input_channels, 
                                 activation=activation)

    def forward(self, tensor, state):
        
        output, state_tensor = self.cell(tensor, state)
        return output, state_tensor

    
class HadamardProduct(nn.Module):
    """
    https://github.com/KimUyen/ConvLSTM-Pytorch/blob/c7b4bd108335a4d6c7d99c00c263346026186b0b/convlstm.py#L10
    """
    def __init__(self, shape):
        super(HadamardProduct, self).__init__()
        self.weights = nn.Parameter(torch.rand(shape))
        
    def forward(self, x):
        return x*self.weights
    

class ConvLSTMCell(nn.Module):

    def __init__(self, shape=(4, 64, 64), num_features=128, kernel=3,  
                 input_channels=128, forget_bias=1.0, activation=nn.Tanh, 
                 normalize=False, peephole=False):
        """
        input_channels: N in RAE or RPM,
        num_features: output features, again, N in RAE or RPM,
        filter_size: kernel size (3 default),
        normalize: whether to normalize or not (True default),
        shape: (B, H, W)
        """
        super(ConvLSTMCell, self).__init__()
        
        self.input_channels = input_channels
        self.kernel = kernel
        self.padding = self.kernel//2
        self.num_features = num_features
        
        self.in_channels = self.input_channels + self.num_features
        self.out_channels = 4 * self.num_features
        
        self.W = nn.Conv2d(self.in_channels, self.out_channels, 
                           self.kernel, 1, self.padding)
        
        self._normalize = normalize
        self._peephole = peephole
        self._activation = activation()
        self._forget_bias = forget_bias
        
        if not self._normalize:
            self.bias = nn.Parameter(torch.zeros((1, self.out_channels, 1, 1)))
        else:
            self.layer_norm_j = nn.LayerNorm([self.num_features, shape[0], shape[1]])
            self.layer_norm_i = nn.LayerNorm([self.num_features, shape[0], shape[1]])
            self.layer_norm_f = nn.LayerNorm([self.num_features, shape[0], shape[1]])
            self.layer_norm_o = nn.LayerNorm([self.num_features, shape[0], shape[1]])
            self.layer_norm_c = nn.LayerNorm([self.num_features, shape[0], shape[1]])
        
        if self._peephole:
            self.W_ci = HadamardProduct((1, batch_size, num_features, 
                                         image_size//4, image_size//4))
            self.W_cf = HadamardProduct((1, batch_size, num_features, 
                                         image_size//4, image_size//4))
            self.W_co = HadamardProduct((1, batch_size, num_features, 
                                         image_size//4, image_size//4))
    
    def forward(self, x, hidden_state):
        h, c = torch.chunk(hidden_state, 2, dim=1)
        h = torch.squeeze(h, dim=1)
        c = torch.squeeze(c, dim=1)
        
        x = torch.cat((x, h), dim=1)
        
        y = self.W(x)
        
        if not self._normalize:
            # Shape not matching, had to use expand_as, each channel has a single bias
            bias = self.bias.expand_as(y)
            y += bias
        
        j, i, f, o = torch.chunk(y, 4, dim=1)
        
        if self._peephole:
            # Hadamard product
            i += self.W_ci(c)
            f += self.W_cf(c)
            
        if self._normalize:
            # Layer normalization
            j = self.layer_norm_j(j)
            i = self.layer_norm_i(i)
            f = self.layer_norm_f(f)
       
        f = F.sigmoid(f + self._forget_bias)
        i = F.sigmoid(i)
        c = c * f + i * self._activation(j)
        
        if self._peephole:
            # Hadamard product
            o += self.W_co(c)
            
        if self._normalize:
            # Layer normalization
            o = self.layer_norm_o(o)
            c = self.layer_norm_c(c)
        
        o = F.sigmoid(o)
        h = o * self._activation(c)
        
        h_clstm = torch.stack((h, c), dim=1)
        
        return h, h_clstm
    

class Bitparm(nn.Module):
    '''
    save params, "END-TO-END OPTIMIZED IMAGE COMPRESSION"
    https://github.com/liujiaheng/iclr_17_compression/blob/bce72bc53f5c5f5da14a54a51eedde5eb8d0c39c/models/bitEstimator.py#L27
    '''

    def __init__(self, channel, final=False):
        super(Bitparm, self).__init__()
        self.final = final
        self.h = nn.Parameter(torch.nn.init.normal_(torch.empty(channel).view(1, -1, 1, 1), 0, 0.01))
        self.b = nn.Parameter(torch.nn.init.normal_(torch.empty(channel).view(1, -1, 1, 1), 0, 0.01))
        if not final:
            self.a = nn.Parameter(torch.nn.init.normal_(torch.empty(channel).view(1, -1, 1, 1), 0, 0.01))
        else:
            self.a = None

    def forward(self, x):
        if self.final:
            return torch.sigmoid(x * F.softplus(self.h) + self.b)
        else:
            x = x * F.softplus(self.h) + self.b
            return x + torch.tanh(x) * torch.tanh(self.a)


class BitEstimator(nn.Module):
    '''
    Estimate bit, Balle et al. "END-TO-END OPTIMIZED IMAGE COMPRESSION"
    https://github.com/liujiaheng/iclr_17_compression/blob/bce72bc53f5c5f5da14a54a51eedde5eb8d0c39c/models/bitEstimator.py#L27
    '''

    def __init__(self, channel):
        super(BitEstimator, self).__init__()
        self.f1 = Bitparm(channel)
        self.f2 = Bitparm(channel)
        self.f3 = Bitparm(channel)
        self.f4 = Bitparm(channel, True)

    def forward(self, x):
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        return self.f4(x)
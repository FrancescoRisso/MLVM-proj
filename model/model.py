import tensorflow as tf
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- CNN Blocks ----------
"""
Args:
    channels: list of channels for each layer
    ks: kernel size
    s: stride
    p: padding
"""
def get_conv_net(channels, ks, s, p):
    layers = []
    for i in range(len(channels) - 1):
        layers.append(nn.Conv2d(channels[i], channels[i+1], kernel_size=ks, stride=s, padding=p))
        layers.append(nn.BatchNorm2d(channels[i+1]))
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)



# ---------- Full Model ----------
class HarmonicCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Yp branch
        self.block_a1 = get_conv_net([3, 16], ks=(5, 5), s=(1, 1), p=(2, 2))
        self.block_a2 = get_conv_net([16, 8], ks=(3, 39), s=(1, 1), p=(1, 19))  # keep time dim
        self.conv_a3 = nn.Conv2d(8, 1, kernel_size=(5, 5), padding=(2, 2))
        self.out_yp = nn.Sigmoid()

        # Yo branch
        self.block_b1 = get_conv_net([3, 32], ks=(5, 5), s=(1, 3), p=(2, 2))
        self.conv_b2 = nn.Conv2d(32, 1, kernel_size=(3, 3), padding=(1, 1))
        self.out_yo = nn.Sigmoid()

        # Yn branch
        self.conv_c1 = nn.Conv2d(9, 1, kernel_size=(7, 3), padding=(3, 1))  # assuming concat across channel
        self.relu_c2 = nn.ReLU()
        self.conv_c3 = nn.Conv2d(1, 32, kernel_size=(7, 7), stride=(1, 3), padding=(3, 3))
        self.conv_c4 = nn.Conv2d(32, 1, kernel_size=(7, 3), padding=(3, 1))
        self.out_yn = nn.Sigmoid()


    # x: (1, 3, H, W)
    def forward(self, x):
        # Yp branch
        xa = self.block_a1(x)
        xa = self.block_a2(xa)
        yp = self.out_yp(self.conv_a3(xa))

        # Yo branch
        xb = self.block_b1(x)
        yo = self.out_yo(self.conv_b2(xb))

        # Concatenate along channel dimension
        concat = torch.cat([xa, xb], dim=1)  # assuming same spatial dims

        # Yn branch
        yn = self.relu_c2(self.conv_c1(concat))
        yn = self.conv_c3(yn)
        yn = self.out_yn(self.conv_c4(yn))

        return yo, yp, yn

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.postprocessing import postprocess
from model.preprocessing import preprocess

# ---------- CNN Blocks ----------
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
        # Yp branch (non viene restituito come output finale)
        self.block_a1 = get_conv_net([3, 16], ks=(5, 5), s=(1, 1), p=(2, 2))
        self.block_a2 = get_conv_net([16, 8], ks=(3, 39), s=(1, 1), p=(1, 19))
        self.conv_a3 = nn.Conv2d(8, 1, kernel_size=(5, 5), padding=(2, 2))
        self.out_yp = nn.Sigmoid()

        # Yo branch (usa output di block_b1 e yn)
        self.block_b1 = get_conv_net([3, 32], ks=(5, 5), s=(1, 1), p=(2, 2))  # stride fixato
        self.conv_b2 = nn.Conv2d(33, 1, kernel_size=(3, 3), padding=(1, 1))
        self.out_yo = nn.Sigmoid()

        # Yn branch (derivata da yp)
        self.conv_c1 = nn.Conv2d(1, 32, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))  # stride fixato
        self.relu_c2 = nn.ReLU()
        self.conv_c3 = nn.Conv2d(32, 1, kernel_size=(7, 3), padding=(3, 1))
        self.out_yn = nn.Sigmoid()

    def forward(self, x):
        x = preprocess(x)  # Assicurati che questa funzione non alteri le dimensioni temporali

        # Yp branch (solo interna)
        xa = self.block_a1(x)
        xa = self.block_a2(xa)
        yp = self.out_yp(self.conv_a3(xa))  # Non restituito

        # Yn branch
        yn = self.relu_c2(self.conv_c1(yp))
        yn = self.out_yn(self.conv_c3(yn))

        # Yo branch (concatena output di block_b1 con yn)
        xb = self.block_b1(x)
        concat = torch.cat([xb, yn], dim=1)
        yo = self.out_yo(self.conv_b2(concat))

        return (yo, yn) #, postprocess(yp, yn, yo)  # Restituisce anche yp per il calcolo della loss

import torch
import torch.nn as nn
from model.postprocessing import postprocess
from model.preprocessing import preprocess
from settings import Settings as s


# ---------- CNN Blocks ----------
def get_conv_net(channels, ks, s, p):
    layers = []
    for i in range(len(channels) - 1):
        layers.append(
            nn.Conv2d(channels[i], channels[i + 1], kernel_size=ks, stride=s, padding=p)
        )
        layers.append(nn.BatchNorm2d(channels[i + 1]))
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)


# ---------- Full Model ----------
class HarmonicCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Yp branch
        self.block_a1 = get_conv_net([3, 16], ks=(5, 5), s=(1, 1), p=(2, 2))
        self.block_a2 = get_conv_net([16, 8], ks=(3, 39), s=(1, 1), p=(1, 19))
        self.conv_a3 = nn.Conv2d(8, 1, kernel_size=(5, 5), padding=(2, 2))

        # Yn branch (uses yp as input)
        self.conv_c1 = nn.Conv2d(
            1, 32, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3)
        )
        self.relu_c2 = nn.ReLU()
        self.conv_c3 = nn.Conv2d(32, 1, kernel_size=(7, 3), padding=(3, 1))

        # Yo branch (uses xb + yn as input)
        self.block_b1 = get_conv_net([3, 32], ks=(5, 5), s=(1, 1), p=(2, 2))
        self.conv_b2 = nn.Conv2d(33, 1, kernel_size=(3, 3), padding=(1, 1))

    def forward(self, x):
        x = preprocess(x)

        # --- Yp branch ---
        xa = self.block_a1(x)
        xa = self.block_a2(xa)
        yp_logits = self.conv_a3(xa)  # logits
        yp = torch.sigmoid(yp_logits)  # only for internal use

        # --- Yn branch ---
        if not s.remove_yn:
            yn_logits = self.conv_c3(self.relu_c2(self.conv_c1(yp)))  # logits
            yn = torch.sigmoid(yn_logits)  # solo per yo
        else:
            yn_logits = None

        # --- Yo branch ---
        xb = self.block_b1(x)
        if s.remove_yn:
            concat = torch.cat([xb, yp], dim=1)
        else:
            concat = torch.cat([xb, yn], dim=1)  # uses "activated" yn output
        yo_logits = self.conv_b2(concat)  # logits

        return (
            yo_logits,
            yp_logits,          # return yp_logits instead of yn_logits
            yn_logits
        )  # postprocessing(Y0, YN) TODO da usare sigmoide anche prima

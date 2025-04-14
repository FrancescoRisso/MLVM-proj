import tensorflow as tf
import numpy as np
import librosa
import torch
import torch.nn as nn

"""
This function computes the Constant-Q Transform (CQT) of an audio file.
Args:
    wav_file (str): Path to the audio file.
    sr (int): Sample rate for loading the audio file.
    hop_length (int): Hop length for the CQT.
    n_bins (int): Number of bins for the CQT.
Returns:
    tf.Tensor: CQT tensor of shape (n_bins, time).
"""
def constant_q_transform(wav_file: str, sr: int, hop_length: int, n_bins: int) -> tf.Tensor:
    # Load the audio file
    y, sr = librosa.load(wav_file, sr=sr)

    # Compute the Constant-Q Transform
    cqt = librosa.cqt(y, sr=sr, hop_length=hop_length, n_bins=n_bins)

    # Convert to TensorFlow tensor
    cqt_tensor = tf.convert_to_tensor(np.abs(cqt), dtype=tf.float32)

    return cqt_tensor

"""
Performs harmonic stacking on a CQT tensor.
Args:
    cqt_tensor (tf.Tensor): CQT tensor of shape (n_bins, time)
    shifts (list[int]): List of harmonic shifts in semitones (e.g., [-12, 0, +12])
Returns:
    tf.Tensor: Stacked tensor of shape (len(shifts), n_bins, time)
"""
def harmonic_stacking(cqt_tensor: tf.Tensor, shifts: list[int]) -> tf.Tensor:
    stacked = []
    cqt_np = cqt_tensor.numpy()

    for shift in shifts:
        shifted = np.roll(cqt_np, shift, axis=0)

        # Se rolla verso l’alto o il basso, zeriamo le bande "wrap-around"
        if shift > 0:
            shifted[:shift, :] = 0
        elif shift < 0:
            shifted[shift:, :] = 0

        stacked.append(shifted)

    stacked_tensor = tf.convert_to_tensor(np.stack(stacked, axis=0), dtype=tf.float32)
    return stacked_tensor

def get_conv_net(channels: list[int], ks: tuple[int, int], s: tuple[int, int], p: tuple[int, int]) -> nn.Module:
    layers = []
    for i in range(len(channels) - 1):
        layers.append(nn.Conv2d(in_channels=channels[i], out_channels=channels[i+1],
                                kernel_size=ks, stride=s, padding=p))
        layers.append(nn.BatchNorm2d(channels[i+1]))
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)



def model(stacked_cqt: tf.Tensor) -> torch.Tensor:
    # Convert TF tensor to PyTorch and reshape
    spectrogram = torch.tensor(stacked_cqt.numpy(), dtype=torch.float32).unsqueeze(0)  # (1, C, H, W)

    # Convolutional network
    conv1 = get_conv_net([1, 16], ks=(5, 5), s=(1, 1), p=(1, 1))
    conv2 = get_conv_net([16, 8], ks=(3, 39), s=(1, 1), p=(1, 1))
    conv3 = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=(5, 5), stride=(1, 1), padding=(1, 1))

    conv_net = nn.Sequential(conv1, conv2, conv3)

    output = conv_net(spectrogram)
    return output

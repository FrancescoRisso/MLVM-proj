import mido
import librosa
import numpy as np
import torch
import pretty_midi
import matplotlib.pyplot as plt

from train.train import train


def main():
    # Esegui il training del modello
    train()


if __name__ == "__main__":
    main()

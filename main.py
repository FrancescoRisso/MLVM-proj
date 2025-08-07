from train.utils import evaluate_and_plot_extremes
from dataloader.split import Split
from train.train import train
import torch
import librosa
from model.model import HarmonicCNN
from model_rnn.model import HarmonicRNN
from settings import Model
from settings import Settings as s
from train.quality_index import evaluate_note_prediction


def main():
    train()
    evaluate_and_plot_extremes("model_saves/harmoniccnn.pth", Split.SINGLE_AUDIO)


if __name__ == "__main__":
    main()

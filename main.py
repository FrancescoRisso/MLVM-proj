from train.train import train
from model import postprocessing
from settings import Settings as s
import torch


def main():
	
    # train()
    postprocessing.model_eval("model_saves/best_model.pth","trial_audio/Midicut.mp3")


if __name__ == "__main__":
    main()

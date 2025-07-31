from train.train import train
from model import postprocessing
from settings import Settings as s


def main():
	
    train()
    postprocessing.model_eval("model_saves/harmoniccnn.pth","trial_audio/shouldbesame.wav")


if __name__ == "__main__":
    main()

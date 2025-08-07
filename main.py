from train.utils import evaluate_and_plot_extremes
from dataloader.split import Split
from train.train import train


def main():
    train()
    evaluate_and_plot_extremes("model_saves/harmoniccnn.pth", Split.SINGLE_AUDIO)


if __name__ == "__main__":
    main()

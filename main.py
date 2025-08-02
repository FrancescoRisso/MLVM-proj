from train.train import evaluate_and_plot_extremes
from dataloader.split import Split
from train.train import train

def main():
    train()
    evaluate_and_plot_extremes("model_saves/best_model.pth", Split.VALIDATION)


if __name__ == "__main__":
    main()

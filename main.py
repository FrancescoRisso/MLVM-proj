from dataloader.split import Split
from train.evaluate import evaluate
from train.train import train


def main():
    import torch

    torch.autograd.set_detect_anomaly(True)
    # Esegui il training del modello
    train()
    
    evaluate("./model_saves/training_RNN/harmoniccnn_trained.pth", Split.VALIDATION)


if __name__ == "__main__":
    main()

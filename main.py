from train.train import train
from settings import Settings as s

def main():
    import torch

    torch.autograd.set_detect_anomaly(True)
    # Esegui il training del modello
    train()
    

if __name__ == "__main__":
    main()

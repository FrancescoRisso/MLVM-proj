from train.train import train


def main():
    import torch

    torch.autograd.set_detect_anomaly(True)
    # Esegui il training del modello
    train()
    

if __name__ == "__main__":
    main()

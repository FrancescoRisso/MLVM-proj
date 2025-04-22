import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import HarmonicCNN
from losses import harmoniccnn_loss  
from dataset import MyDataset


# ---------- Training Function for one epoch ----------
def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    label_smoothing: float = 0.1,
    weighted: bool = True,
    positive_weight: float = 0.95,
    show_progress: bool = True
) -> float:
    
    model.train()
    running_loss = 0.0
    total_batches = len(dataloader)

    for batch in (tqdm(dataloader) if show_progress else dataloader):
        # Get these from the dataloeder (get item from dataset)
        x = batch["input"].to(device) 
        yo_true = batch["yo"].to(device)   
        yp_true = batch["yp"].to(device)
        yn_true = batch["yn"].to(device)

        optimizer.zero_grad()
        yo_pred, yp_pred, yn_pred = model(x)

        loss = harmoniccnn_loss(
            yo_pred, yp_pred, yn_pred,
            yo_true, yp_true, yn_true,
            label_smoothing=label_smoothing,
            weighted=weighted,
            positive_weight=positive_weight
        )

        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    return running_loss / total_batches


# ---------- Training function ----------
def train():

    epochs = 10
    batch_size = 16
    learning_rate = 1e-4
    label_smoothing = 0.1
    weighted = True
    positive_weight = 0.95
    weighted = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    model = HarmonicCNN().to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_dataset = MyDataset(split="train")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


    for epoch in range(epochs):
        avg_loss = train_one_epoch(
            model, train_loader, optimizer, device,
            label_smoothing=label_smoothing,
            weighted=weighted,
            positive_weight=positive_weight
        )
        print(f"[Epoch {epoch+1}/{epochs}] Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "harmoniccnn_trained.pth")
    print("Model saved as 'harmoniccnn_trained.pth'")


if __name__ == "__main__":
    train()

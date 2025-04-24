import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model.model import HarmonicCNN
from train.losses import harmoniccnn_loss  
from dataloader.dataset import DataSet 
from dataloader.split import Split
from settings import Settings as s
import pretty_midi
import librosa
import numpy as np

def midi_to_label_matrices(midi_path: str, sr: int, hop_length: int, n_bins: int, fmin=librosa.note_to_hz('C1')) -> dict:
    midi = pretty_midi.PrettyMIDI(midi_path)
    duration = midi.get_end_time()
    n_frames = int(np.ceil(duration * sr / hop_length))

    # Frequenze corrispondenti alle bin
    freqs = librosa.cqt_frequencies(n_bins=n_bins, fmin=fmin)

    yo = np.zeros((n_bins, n_frames), dtype=np.float32)
    yp = np.zeros((n_bins, n_frames), dtype=np.float32)
    yn = np.zeros((n_bins, n_frames), dtype=np.float32)

    for note in midi.instruments[0].notes:
        start_frame = int(note.start * sr / hop_length)
        end_frame = int(note.end * sr / hop_length)
        pitch_hz = librosa.midi_to_hz(note.pitch)

        # Trovare bin più vicino
        bin_index = np.argmin(np.abs(freqs - pitch_hz))
        if bin_index < 0 or bin_index >= n_bins:
            continue

        if start_frame < n_frames:
            yo[bin_index, start_frame] = 1.0  # onset
        yn[bin_index, start_frame:end_frame] = 1.0  # note
        yp[bin_index, start_frame:end_frame] = 1.0  # contour

    return {
        "yo": torch.tensor(yo),
        "yn": torch.tensor(yn),
        "yp": torch.tensor(yp),
    }


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

    for batch in (dataloader):
        # Get these from the dataloeder (get item from dataset)
        
        midis, audios = batch  #tutti i midi e tutti gli audio del batch
        
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


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    model = HarmonicCNN().to(device)

    optimizer = optim.Adam(model.parameters(), lr=s.learning_rate)

    train_dataset = DataSet(Split.TRAIN, s.seconds)
    train_loader = DataLoader(train_dataset, batch_size=s.batch_size, shuffle=True)


    for epoch in range(s.epochs):
        avg_loss = train_one_epoch(
            model, train_loader, optimizer, device,
            label_smoothing=s.label_smoothing,
            weighted=s.weighted,
            positive_weight=s.positive_weight
        )
        print(f"[Epoch {epoch+1}/{s.epochs}] Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "harmoniccnn_trained.pth")
    print("Model saved as 'harmoniccnn_trained.pth'")


if __name__ == "__main__":
    train()

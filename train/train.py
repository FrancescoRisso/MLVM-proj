import numpy as np
import torch
import torch.optim as optim
import mido

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataloader.dataset import DataSet
from dataloader.Song import Song
from dataloader.split import Split
from model.model import HarmonicCNN
from settings import Settings as s
from train.losses import harmoniccnn_loss


def midi_to_label_matrices(mido_midi, sample_rate, hop_length, n_bins=88):
    ticks_per_beat = mido_midi.ticks_per_beat
    min_pitch = 21 # Pitch corrispondente a "A0"
    max_pitch = min_pitch + n_bins

    # Tempo iniziale
    current_tempo = 500000  # Default 120 BPM
    time_in_seconds = 0.0
    tick_accumulator = 0

    # Lista delle note (pitch, start_time_sec, end_time_sec)
    active_notes = {}
    notes = []

    for msg in mido.merge_tracks(mido_midi.tracks):
        tick_accumulator += msg.time
        if msg.type == 'set_tempo':
            current_tempo = msg.tempo
        time_in_seconds = mido.tick2second(tick_accumulator, ticks_per_beat, current_tempo)

        if msg.type == 'note_on' and msg.velocity > 0:
            active_notes[msg.note] = time_in_seconds
        elif (msg.type == 'note_off') or (msg.type == 'note_on' and msg.velocity == 0):
            start = active_notes.pop(msg.note, None)
            if start is not None and min_pitch <= msg.note < max_pitch:
                notes.append((msg.note, start, time_in_seconds))

    # Determina la durata massima per dimensionare le matrici
    max_time = s.seconds
    n_frames = int(np.ceil(max_time * sample_rate / hop_length))
    yo = np.zeros((n_bins, n_frames), dtype=np.float32)
    yn = np.zeros((n_bins, n_frames), dtype=np.float32)

    for pitch, start, end in notes:
        p = pitch - min_pitch
        start_frame = int(np.floor(start * sample_rate / hop_length))
        end_frame = int(np.ceil(end * sample_rate / hop_length))

        yo[p, start_frame] = 1.0  # nota inizia
        yn[p, start_frame:end_frame] = 1.0  # nota attiva

    return yo, yn

def to_tensor(array):
    return torch.tensor(array) if isinstance(array, np.ndarray) else array

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if isinstance(tensor, torch.Tensor) else tensor

def soft_accuracy(y_pred: torch.Tensor, y_true: torch.Tensor, tolerance: float = 0.1) -> float:
    """
    Calcola l'accuratezza come la percentuale di elementi per cui la predizione
    è entro ± tolerance rispetto al valore vero.

    Args:
        y_pred: output del modello, shape (B, 88, T)
        y_true: ground truth, shape (B, 88, T)
        tolerance: soglia di errore tollerata

    Returns:
        Accuratezza come valore float
    """
    with torch.no_grad():
        diff = torch.abs(y_pred - y_true)
        correct = (diff <= tolerance).float()
        accuracy = correct.mean().item()
    return accuracy

def plot_prediction_vs_ground_truth(yo_pred, yn_pred, yo_true, yn_true):

    yo_pred_np = to_numpy(yo_pred)
    yo_true_np = to_numpy(yo_true)
    yn_pred_np = to_numpy(yn_pred)
    yn_true_np = to_numpy(yn_true)

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.imshow(yo_true_np, aspect='auto', cmap='viridis')
    plt.title("YO Ground Truth")
    plt.colorbar()

    plt.subplot(2, 2, 2)
    plt.imshow(yo_pred_np, aspect='auto', cmap='viridis')
    plt.title("YO Prediction")
    plt.colorbar()

    plt.subplot(2, 2, 3)
    plt.imshow(yn_true_np, aspect='auto', cmap='inferno')
    plt.title("YN Ground Truth")
    plt.colorbar()

    plt.subplot(2, 2, 4)
    plt.imshow(yn_pred_np, aspect='auto', cmap='inferno')
    plt.title("YN Prediction")
    plt.colorbar()

    plt.tight_layout()
    plt.show()

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    running_loss = 0.0
    total_batches = len(dataloader)

    for batch_idx, batch in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}/{total_batches}")
        (midis_np, tempos, ticks_per_beats, nums_messages), audios = batch
        
        yo_true_batch = []
        yn_true_batch = []
        audio_input_batch = []

        for i in range(midis_np.shape[0]):
            midi = Song.from_np(midis_np[i], tempos[i], ticks_per_beats[i], nums_messages[i]).get_midi()
            yo, yn = midi_to_label_matrices(midi, s.sample_rate, s.hop_length, n_bins=88)

            yo_true_batch.append(to_tensor(yo).to(device))  # Aggiungi al batch
            yn_true_batch.append(to_tensor(yn).to(device))  # Aggiungi al batch
            audio_input_batch.append(audios[i].to(device))  # Aggiungi al batch

        # Converti il batch in tensori
        yo_true_batch = torch.stack(yo_true_batch)  # Forma finale: [batch_size, 88, 87]
        yn_true_batch = torch.stack(yn_true_batch)  # Forma finale: [batch_size, 88, 87]
        audio_input_batch = torch.stack(audio_input_batch)  # Forma finale: [batch_size, audio_features]

        #print(f"YO true shape: {yo_true_batch.shape}")
        #print(f"YN true shape: {yn_true_batch.shape}")
        #print(f"Audio input shape: {audio_input_batch.shape}")
        
        optimizer.zero_grad()

        yo_pred, yn_pred = model(audio_input_batch)

        yo_pred = yo_pred.squeeze(1)
        yn_pred = yn_pred.squeeze(1)
        
        #print(f"YO pred shape: {yo_pred.shape}")
        #print(f"YN pred shape: {yn_pred.shape}")
        
        loss = harmoniccnn_loss(
            yo_pred,
            yn_pred,
            yo_true_batch,
            yn_true_batch,
            label_smoothing=s.label_smoothing,
            weighted=s.weighted,
            positive_weight=s.positive_weight,
        )
        
        # "Fake" accuracy
        acc_yo = soft_accuracy(yo_pred, yo_true_batch)
        acc_yn = soft_accuracy(yn_pred, yn_true_batch)
        print(f"Loss: {loss.item():.4f} | YO Accuracy: {acc_yo:.4f} | YN Accuracy: {acc_yn:.4f}")
        
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        print(f"Runing loss: {loss}")

        if batch_idx == 30:
            print("Plotting predictions vs ground truth...")
            plot_prediction_vs_ground_truth(
                yo_pred[0],  
                yn_pred[0],  
                yo_true_batch[0],  
                yn_true_batch[0]  
            )
            
    return running_loss / total_batches


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    model = HarmonicCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=s.learning_rate)

    train_dataset = DataSet(Split.TRAIN, s.seconds)
    train_loader = DataLoader(train_dataset, batch_size=s.batch_size, shuffle=True)

    for epoch in range(s.epochs):
        avg_loss = train_one_epoch(model, train_loader, optimizer, device)
        print(f"[Epoch {epoch+1}/{s.epochs}] Loss: {avg_loss:.4f}")

    #torch.save(model.state_dict(), "harmoniccnn_trained.pth")
    #print("Model saved as 'harmoniccnn_trained.pth'")


if __name__ == "__main__":
    train()

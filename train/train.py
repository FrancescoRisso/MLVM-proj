import os
import torch
import torch.optim as optim
import tqdm

from datetime import datetime
from torch.utils.data import DataLoader
from dataloader.dataset import DataSet
from dataloader.Song import Song
from dataloader.split import Split
from model.model import HarmonicCNN
from settings import Settings as s
from train.losses import harmoniccnn_loss
from train.utils import (
    midi_to_label_matrices,
    to_tensor,
    weighted_soft_accuracy,
    plot_prediction_vs_ground_truth,
    batch_prediction_plot,
)


def train_one_epoch(model, dataloader, optimizer, device, epoch, session_dir):
    model.train()
    running_loss = 0.0
    total_batches = len(dataloader)

    for batch_idx, batch in tqdm.tqdm(
        enumerate(dataloader), total=total_batches, desc=f"Epoch {epoch+1}/{s.epochs}"
    ):

        (midis_np, tempos, ticks_per_beats, nums_messages), audios = batch

        yo_true_batch = []
        yn_true_batch = []
        audio_input_batch = []

        for i in range(midis_np.shape[0]):
            midi = Song.from_np(
                midis_np[i], tempos[i], ticks_per_beats[i], nums_messages[i]
            ).get_midi()
            yo, yn = midi_to_label_matrices(
                midi, s.sample_rate, s.hop_length, n_bins=88
            )

            yo_true_batch.append(to_tensor(yo).to(device))  # Aggiungi al batch
            yn_true_batch.append(to_tensor(yn).to(device))  # Aggiungi al batch
            audio_input_batch.append(audios[i].to(device))  # Aggiungi al batch

        # Converti il batch in tensori
        yo_true_batch = torch.stack(yo_true_batch)  # Forma finale: [batch_size, 88, 87]
        yn_true_batch = torch.stack(yn_true_batch)  # Forma finale: [batch_size, 88, 87]
        audio_input_batch = torch.stack(
            audio_input_batch
        )  # Forma finale: [batch_size, audio_features]

        optimizer.zero_grad()

        (yo_pred, yn_pred) = model(audio_input_batch)

        yo_pred = yo_pred.squeeze(1)
        yn_pred = yn_pred.squeeze(1)

        # calcola le weigehted soft accuracy per debug
        yo_soft_accuracy = weighted_soft_accuracy(
            yo_pred, yo_true_batch, 0.95, 0.05, 0.1
        )
        yn_soft_accuracy = weighted_soft_accuracy(
            yn_pred, yn_true_batch, 0.95, 0.05, 0.1
        )
        print(
            f"yo_soft_accuracy: {yo_soft_accuracy:.4f}, yn_soft_accuracy: {yn_soft_accuracy:.4f}"
        )

        loss = harmoniccnn_loss(
            yo_pred,
            yn_pred,
            yo_true_batch,
            yn_true_batch,
            label_smoothing=s.label_smoothing,
            weighted=s.weighted,
            positive_weight=s.positive_weight,
        )

        # If is the last batch of the last epoch, plot the prediction vs ground truth
        if batch_idx == total_batches - 1 and epoch == s.epochs - 1:
            # Plot the prediction vs ground truth
            plot_prediction_vs_ground_truth(
                yo_pred[0], yn_pred[0], yo_true_batch[0], yn_true_batch[0]
            )

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    if s.save_model:
        os.makedirs(session_dir, exist_ok=True)
        path = os.path.join(session_dir, f"harmoniccnn_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), path)
        print(f"Model saved as '{path}'")

    return running_loss / total_batches


def train():

    device = s.device
    print(f"Training on {device}")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    session_dir = os.path.join(f"model_saves", f"training_{timestamp}")

    model = HarmonicCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=s.learning_rate)

    train_dataset = DataSet(Split.TRAIN, s.seconds)
    train_loader = DataLoader(train_dataset, batch_size=s.batch_size, shuffle=True)

    for epoch in range(s.epochs):
        avg_loss = train_one_epoch(
            model, train_loader, optimizer, device, epoch, session_dir
        )
        print(f"[Epoch {epoch+1}/{s.epochs}] Loss: {avg_loss:.4f}")

    final_model_path = os.path.join(session_dir, "harmoniccnn_trained.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Model saved as '{final_model_path}'")

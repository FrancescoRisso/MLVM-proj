import os
import random
import sys
from datetime import datetime

import torch
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader

from dataloader.dataset import DataSet
from dataloader.Song import Song
from dataloader.split import Split
from model.model import HarmonicCNN
from model_rnn.model import HarmonicRNN
from settings import Model
from settings import Settings as s
from train.losses import harmoniccnn_loss
from train.rnn_losses import np_midi_loss
from train.utils import (
    midi_to_label_matrices,
    plot_prediction_vs_ground_truth,
    to_tensor,
    weighted_soft_accuracy,
)


def train_one_epoch(model: HarmonicCNN | HarmonicRNN, dataloader, optimizer, device, epoch, session_dir):
    model.train()
    running_loss = 0.0
    total_batches = len(dataloader)

    for batch_idx, batch in tqdm.tqdm(
        enumerate(dataloader), total=total_batches, desc=f"Epoch {epoch+1}/{s.epochs}"
    ):

        (midis_np, tempos, ticks_per_beats, nums_messages), audios = batch

        optimizer.zero_grad()

        if s.model == Model.CNN:
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

            (yo_pred, yp_pred, yn_pred) = model(audio_input_batch)

            yo_pred = yo_pred.squeeze(1)
            yp_pred = yp_pred.squeeze(1)
            yn_pred = yn_pred.squeeze(1)

            # calcola le weigehted soft accuracy per debug
            yo_soft_accuracy = weighted_soft_accuracy(
                yo_pred, yo_true_batch, 0.95, 0.05, 0.1
            )
            yn_soft_accuracy = weighted_soft_accuracy(
                yn_pred, yn_true_batch, 0.95, 0.05, 0.1
            )
            yp_soft_accuracy = weighted_soft_accuracy(
                yp_pred, yn_true_batch, 0.95, 0.05, 0.1
            )

            print(
                f"\nyo_soft_accuracy: {yo_soft_accuracy:.4f}, yp_soft_accuracy: {yp_soft_accuracy:.4f}, yn_soft_accuracy: {yn_soft_accuracy:.4f}"
            )

            loss = harmoniccnn_loss(
                yo_pred,
                yp_pred,
                yo_true_batch,
                yn_true_batch,
                yn_pred,
                yn_true_batch,
                label_smoothing=s.label_smoothing,
                weighted=s.weighted,
                positive_weight=s.positive_weight,
            )

            total_loss = sum(loss.values())

            # If is the last batch of the last epoch, plot the prediction vs ground truth
            if batch_idx == total_batches - 1 and epoch == s.epochs - 1:
                # Plot the prediction vs ground truth
                plot_prediction_vs_ground_truth(
                    yo_pred[0], yn_pred[0], yo_true_batch[0], yn_true_batch[0]
                )
        
        else: # Using RNN
            assert isinstance(model, HarmonicRNN)

            audios = audios.reshape((
                audios.shape[0], # leave batch items untouched
                -1, # all the seconds
                s.sample_rate  # samples per secon
            ))

            pred_midi, pred_len = model(audios)

            total_loss = np_midi_loss(pred_midi, pred_len, midis_np, nums_messages)

        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()

    if s.save_model:
        os.makedirs(session_dir, exist_ok=True)
        path = os.path.join(session_dir, f"harmoniccnn_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), path)
        print(f"Model saved as '{path}'")

    return running_loss / total_batches


def train():

    # Print random seed to debug potential errors due to randomness
    seed = random.randrange(sys.maxsize)
    random.seed(seed)
    print("Seed was:", seed)

    device = s.device
    print(f"Training on {device}")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    session_dir = os.path.join(f"model_saves", f"training_{timestamp}")

    model = (HarmonicCNN() if s.model == Model.CNN else HarmonicRNN()).to(device)
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

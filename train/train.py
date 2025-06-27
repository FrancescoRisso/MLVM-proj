import os
import random
import sys
from datetime import datetime

import torch
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader

import wandb

from dataloader.dataset import DataSet
from dataloader.Song import Song
from dataloader.split import Split
from model.model import HarmonicCNN
from model_rnn.model import HarmonicRNN
from settings import Model
from settings import Settings as s
from train.losses import harmoniccnn_loss
from train.rnn_losses import np_midi_loss
from train.evaluate import evaluate
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
            yp_true_batch = []
            
            audio_input_batch = []

            for i in range(midis_np.shape[0]):
                midi = Song.from_np(
                    midis_np[i], tempos[i], ticks_per_beats[i], nums_messages[i]
                ).get_midi()
                yo, yp = midi_to_label_matrices(
                    midi, s.sample_rate, s.hop_length, n_bins=88
                )
                yn = yp

                yo_true_batch.append(to_tensor(yo).to(device))
                yp_true_batch.append(to_tensor(yp).to(device))
                if not s.remove_yn:
                    yn_true_batch.append(to_tensor(yn).to(device))
                    
                audio_input_batch.append(audios[i].to(device))

            yo_true_batch = torch.stack(yo_true_batch)
            yp_true_batch = torch.stack(yp_true_batch)
            if not s.remove_yn:
                yn_true_batch = torch.stack(yn_true_batch)

            audio_input_batch = torch.stack(audio_input_batch)

            (yo_pred, yp_pred, yn_pred) = model(audio_input_batch)
            yo_pred = yo_pred.squeeze(1)
            yp_pred = yp_pred.squeeze(1)
            if not s.remove_yn:
                yn_pred = yn_pred.squeeze(1)
            
            # TODO MIGLIORARE IL CALCOLO DELLA SOFT ACCURACY
            
            # yo_soft_accuracy = weighted_soft_accuracy(
            #     yo_pred, yo_true_batch, 0.95, 0.05, 0.1
            # )
            # yp_soft_accuracy = weighted_soft_accuracy(
            #     yn_pred, yn_true_batch, 0.95, 0.05, 0.1
            # )
            # print(
            #     f"yo_soft_accuracy: {yo_soft_accuracy:.4f}, yp_soft_accuracy: {yp_soft_accuracy:.4f}"
            # )
            
            if not s.remove_yn:
                loss = harmoniccnn_loss(
                    yo_pred,       # yo_logits
                    yp_pred,       # yp_logits
                    yo_true_batch,  # yo_true
                    yp_true_batch,  # yp_true
                    yn_pred,        # yn_logits (opzionale)
                    yn_true_batch,  # yn_true (opzionale)
                    label_smoothing=s.label_smoothing,
                    weighted=s.weighted,
                    positive_weight=s.positive_weight,
                )
            else:
                loss = harmoniccnn_loss(
                    yo_pred,        # yo_logits
                    yp_pred,        # yp_logits
                    yo_true_batch,   # yo_true
                    yp_true_batch,   # yp_true
                    # yn_logits e yn_true omessi
                    label_smoothing=s.label_smoothing,
                    weighted=s.weighted,
                    positive_weight=s.positive_weight,
                )
                
            total_loss = sum(loss.values())

            # TODO PLOTTA SUL SITO NON IN CONSOLE
            
            # if batch_idx == total_batches - 1 and epoch == s.epochs - 1:
            #     if not s.remove_yn:
            #         plot_prediction_vs_ground_truth(
            #             yo_pred[0], yp_pred[0], yo_true_batch[0], yp_true_batch[0]
            #         )
            #     else:
            #         plot_prediction_vs_ground_truth(
            #             yo_pred[0], yp_pred[0], yo_true_batch[0], yp_true_batch[0]
            #         )
        else:
            assert isinstance(model, HarmonicRNN)
            audios = audios.reshape((
                audios.shape[0],
                -1,
                s.sample_rate
            ))

            pred_midi, pred_len = model(audios)
            total_loss = np_midi_loss(pred_midi, pred_len, midis_np, nums_messages)


        total_loss.backward()
        optimizer.step()
        running_loss += total_loss.item()

    if s.save_model:
        os.makedirs(session_dir, exist_ok=True)
        if s.model == Model.CNN:
            path = os.path.join(session_dir, "harmoniccnn.pth")
        else:
            path = os.path.join(session_dir, "harmonicrnn.pth")
            
        torch.save(model.state_dict(), path)
        print(f"Model saved as '{path}'")


    return running_loss / total_batches


def train():
    seed = random.randrange(sys.maxsize)
    random.seed(seed)
    print("Seed was:", seed)

    device = s.device
    print(f"Training on {device}")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    session_dir = os.path.join("model_saves")

    wandb.init(
        project="MLVM-Project",
        name=f"harmonic_training_{timestamp}",
        config={
            "epochs": s.epochs,
            "batch_size": s.batch_size,
            "learning_rate": s.learning_rate,
            "model": s.model.name,
            "seed": seed,
        }
    )

    model = (HarmonicCNN() if s.model == Model.CNN else HarmonicRNN()).to(device)
    if s.pre_trained_model_path != None:
        model.load_state_dict(
            torch.load(s.pre_trained_model_path, map_location=device)
        )
        print(f"Loaded pre-trained model from {s.pre_trained_model_path}")

    optimizer = optim.Adam(model.parameters(), lr=s.learning_rate)

    train_dataset = DataSet(Split.TRAIN, s.seconds)
    train_loader = DataLoader(train_dataset, batch_size=s.batch_size, shuffle=True)

    # Early Stopping Setup
    best_val_loss = float("inf")
    patience = 5
    patience_counter = 0

    for epoch in range(s.epochs):
        # Training
        avg_train_loss = train_one_epoch(
            model, train_loader, optimizer, device, epoch, session_dir
        )
        print(f"[Epoch {epoch+1}/{s.epochs}] Train Loss: {avg_train_loss:.4f}")

        wandb.log({
            "epoch": epoch + 1,
            "train/loss": avg_train_loss,
        })

        # Evaluation (Validation)
        if s.model == Model.CNN:
            model_path = os.path.join(session_dir, "harmoniccnn.pth")
        else:
            model_path = os.path.join(session_dir, "harmonicrnn.pth")

        print("Evaluating on validation set...")
        avg_val_loss = evaluate(model_path, Split.VALIDATION)
        print(f"[Epoch {epoch+1}/{s.epochs}] Validation Loss: {avg_val_loss:.4f}")

        wandb.log({
            "epoch": epoch + 1,
            "val/loss": avg_val_loss,
        })

        # Early Stopping Logic
        if avg_val_loss < best_val_loss - 1e-4:
            best_val_loss = avg_val_loss
            patience_counter = 0
            print(f"Validation loss improved to {best_val_loss:.4f}")
            
            # Save best model checkpoint
            best_model_path = os.path.join(session_dir, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved to {best_model_path}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}. Best val loss: {best_val_loss:.4f}")
                break

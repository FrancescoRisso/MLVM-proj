import numpy as np
import torch

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
    soft_accuracy,
    plot_prediction_vs_ground_truth,
)


def evaluate(model_path):
    device = s.device
    print(f"Evaluating on {device}")

    model = HarmonicCNN().to(device)

    if model_path is not None:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded pre-trained model from {model_path}")
    else:
        print("No model found insert a valid model")
        return

    model.eval()

    test_dataset = DataSet(Split.TEST, s.seconds)
    test_loader = DataLoader(test_dataset, batch_size=s.batch_size, shuffle=False)

    running_loss = 0.0
    total_batches = len(test_loader)

    all_acc_yo = []
    all_acc_yn = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
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

                yo_true_batch.append(to_tensor(yo).to(device))
                yn_true_batch.append(to_tensor(yn).to(device))
                audio_input_batch.append(audios[i].to(device))

            yo_true_batch = torch.stack(yo_true_batch)
            yn_true_batch = torch.stack(yn_true_batch)
            audio_input_batch = torch.stack(audio_input_batch)

            yo_pred, yn_pred = model(audio_input_batch)
            yo_pred = yo_pred.squeeze(1)
            yn_pred = yn_pred.squeeze(1)

            loss = harmoniccnn_loss(
                yo_pred,
                yn_pred,
                yo_true_batch,
                yn_true_batch,
                label_smoothing=s.label_smoothing,
                weighted=s.weighted,
                positive_weight=s.positive_weight,
            )

            acc_yo = soft_accuracy(yo_pred, yo_true_batch)
            acc_yn = soft_accuracy(yn_pred, yn_true_batch)

            running_loss += loss.item()
            all_acc_yo.append(acc_yo)
            all_acc_yn.append(acc_yn)

            if batch_idx == 0:
                print("Plotting predictions vs ground truth (first batch)...")
                plot_prediction_vs_ground_truth(
                    yo_pred[0], yn_pred[0], yo_true_batch[0], yn_true_batch[0]
                )

    avg_loss = running_loss / total_batches
    avg_acc_yo = np.mean(all_acc_yo)
    avg_acc_yn = np.mean(all_acc_yn)

    print(
        f"[Evaluation] Loss: {avg_loss:.4f} | YO Acc: {avg_acc_yo:.4f} | YN Acc: {avg_acc_yn:.4f}"
    )

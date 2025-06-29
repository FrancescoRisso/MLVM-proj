import numpy as np
import torch
import mido
import matplotlib.pyplot as plt
import torch.nn.functional as F

from sklearn.metrics import f1_score
from dataloader.Song import Song
from settings import Settings as s
from train.losses import harmoniccnn_loss


def midi_to_label_matrices(mido_midi, sample_rate, hop_length, n_bins=88):
    ticks_per_beat = mido_midi.ticks_per_beat
    min_pitch = 21  # Pitch corrispondente a "A0"
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
        if msg.type == "set_tempo":
            current_tempo = msg.tempo
        time_in_seconds = mido.tick2second(
            tick_accumulator, ticks_per_beat, current_tempo
        )

        if msg.type == "note_on" and msg.velocity > 0:
            active_notes[msg.note] = time_in_seconds
        elif (msg.type == "note_off") or (msg.type == "note_on" and msg.velocity == 0):
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
    if tensor is None:
        return None
    return tensor.detach().cpu().numpy() if isinstance(tensor, torch.Tensor) else tensor


def soft_continuous_accuracy(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    Accuracy continua: 1 - |pred - true| mediato.
    Valori predetti vicini al target sono premiati di più.
    """
    with torch.no_grad():
        error = torch.abs(y_pred - y_true)
        score = 1.0 - error
        return score.mean().item()


def binary_classification_metrics(
    y_pred: torch.Tensor, y_true: torch.Tensor, threshold: float = 0.5
):
    with torch.no_grad():
        y_pred_bin = (y_pred >= threshold).float()

        tp = (y_pred_bin * y_true).sum().item()
        fp = (y_pred_bin * (1 - y_true)).sum().item()
        fn = ((1 - y_pred_bin) * y_true).sum().item()
        tn = ((1 - y_pred_bin) * (1 - y_true)).sum().item()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        accuracy = soft_continuous_accuracy(y_pred, y_true)

        return {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Accuracy": accuracy,
        }


def plot_prediction_vs_ground_truth(
    yo_pred, yp_pred, yn_pred, yo_true, yp_true, yn_true
):
    yo_pred_np = to_numpy(yo_pred)
    yp_pred_np = to_numpy(yp_pred)
    yn_pred_np = None if s.remove_yn else to_numpy(yn_pred)

    yo_true_np = to_numpy(yo_true)
    yp_true_np = to_numpy(yp_true)
    yn_true_np = None if s.remove_yn else to_numpy(yn_true)

    # Numero di righe: 2 per YO e YP, +1 se consideri YN
    n_rows = 2 if s.remove_yn else 3
    n_cols = 2  # sempre 2 colonne: ground truth e predizione

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))

    # Plot YO
    im0 = axes[0, 0].imshow(yo_true_np, aspect="auto", origin="lower")
    axes[0, 0].set_title("YO Ground Truth")
    fig.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].imshow(yo_pred_np, aspect="auto", origin="lower")
    axes[0, 1].set_title("YO Prediction")
    fig.colorbar(im1, ax=axes[0, 1])

    # Plot YP
    im2 = axes[1, 0].imshow(yp_true_np, aspect="auto", origin="lower")
    axes[1, 0].set_title("YP Ground Truth")
    fig.colorbar(im2, ax=axes[1, 0])

    im3 = axes[1, 1].imshow(yp_pred_np, aspect="auto", origin="lower")
    axes[1, 1].set_title("YP Prediction")
    fig.colorbar(im3, ax=axes[1, 1])

    # Plot YN (solo se non rimosso)
    if not s.remove_yn:
        im4 = axes[2, 0].imshow(yn_true_np, aspect="auto", origin="lower")
        axes[2, 0].set_title("YN Ground Truth")
        fig.colorbar(im4, ax=axes[2, 0])

        im5 = axes[2, 1].imshow(yn_pred_np, aspect="auto", origin="lower")
        axes[2, 1].set_title("YN Prediction")
        fig.colorbar(im5, ax=axes[2, 1])

    for ax_row in axes:
        for ax in ax_row:
            ax.axis("off")

    plt.tight_layout()
    return fig


def should_log_image(epoch):
    # Logga ogni 2 epoche per le prime 10, poi ogni 5 epoche
    if epoch <= 10:
        return epoch % 2 == 0
    else:
        return epoch % 5 == 0


def binary_classification_metrics(
    y_pred: torch.Tensor, y_true: torch.Tensor, threshold: float = 0.5
):
    """
    Calcola TP, FP, FN, TN, Precision, Recall e F1 score per predizioni binarie.

    y_pred: tensor dopo la sigmoid, valori in [0,1]
    y_true: tensor binario (0/1)
    """
    with torch.no_grad():
        y_pred_bin = (y_pred >= threshold).float()

        tp = (y_pred_bin * y_true).sum().item()
        fp = (y_pred_bin * (1 - y_true)).sum().item()
        fn = ((1 - y_pred_bin) * y_true).sum().item()
        tn = ((1 - y_pred_bin) * (1 - y_true)).sum().item()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        return {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
        }

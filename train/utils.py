import numpy as np
import torch
import mido
import matplotlib.pyplot as plt

from dataloader.Song import Song
from settings import Settings as s


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
    Soft continuous accuracy: 1 - |pred - true| averaged.
    Predicted values closer to the target are rewarded more.
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


def imshow_fixed(ax, data, title, fig):
    data = np.squeeze(data)
    im = ax.imshow(data, aspect="auto", origin="lower", vmin=0, vmax=1)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)


def plot_harmoniccnn_outputs(yo, yp, yn, title_prefix):

    yo_np = to_numpy(yo)
    yp_np = to_numpy(yp)
    yn_np = None if s.remove_yn else to_numpy(yn)

    n_rows = 2 if s.remove_yn else 3
    fig, axes = plt.subplots(n_rows, 1, figsize=(6, 4 * n_rows))

    if n_rows == 1:
        axes = [axes]
    axes = np.atleast_1d(axes)

    imshow_fixed(axes[0], yo_np, f"YO {title_prefix}", fig)
    imshow_fixed(axes[1], yp_np, f"YP {title_prefix}", fig)

    if not s.remove_yn:
        imshow_fixed(axes[2], yn_np, f"YN {title_prefix}", fig)

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    return fig


def should_log_image(epoch):
    if epoch < 10:
        return epoch % 2 == 0
    else:
        return epoch % 5 == 0


def binary_classification_metrics(
    y_pred: torch.Tensor, y_true: torch.Tensor, threshold: float = 0.5
):
    """
    Computes TP, FP, FN, TN, Precision, Recall, and F1 score for binary predictions.

    y_pred: tensor after sigmoid, values in [0,1]
    y_true: binary tensor (0/1)
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


@torch.no_grad()
def plot_fixed_sample(model, sample, device):
    (midi_np, tempo, ticks_per_beat, num_messages), audio = sample
    audio = torch.from_numpy(audio) if isinstance(audio, np.ndarray) else audio
    audio = audio.unsqueeze(0).to(device)

    midi = Song.from_np(midi_np, tempo, ticks_per_beat, num_messages).get_midi()
    yo_true, yp_true = midi_to_label_matrices(
        midi, s.sample_rate, s.hop_length, n_bins=88
    )
    yn_true = yp_true if s.remove_yn else yo_true

    yo_true = to_tensor(yo_true).unsqueeze(0).to(device)
    yp_true = to_tensor(yp_true).unsqueeze(0).to(device)
    yn_true = to_tensor(yn_true).unsqueeze(0).to(device) if not s.remove_yn else None

    yo_pred, yp_pred, yn_pred = model(audio)

    yo_pred = torch.sigmoid(yo_pred).squeeze(1).cpu()
    yp_pred = torch.sigmoid(yp_pred).squeeze(1).cpu()
    yn_pred = torch.sigmoid(yn_pred).squeeze(1).cpu() if not s.remove_yn else None

    title_prefix = "Prediction"

    fig = plot_harmoniccnn_outputs(yo_pred, yp_pred, yn_pred, title_prefix)

    return fig

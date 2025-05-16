import numpy as np
import torch
import mido
import matplotlib.pyplot as plt

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
    return tensor.detach().cpu().numpy() if isinstance(tensor, torch.Tensor) else tensor


def soft_accuracy(
    y_pred: torch.Tensor, y_true: torch.Tensor, tolerance: float = 0.1
) -> float:
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
    plt.imshow(yo_true_np, aspect="auto", cmap="viridis")
    plt.title("YO Ground Truth")
    plt.colorbar()

    plt.subplot(2, 2, 2)
    plt.imshow(yo_pred_np, aspect="auto", cmap="viridis")
    plt.title("YO Prediction")
    plt.colorbar()

    plt.subplot(2, 2, 3)
    plt.imshow(yn_true_np, aspect="auto", cmap="inferno")
    plt.title("YN Ground Truth")
    plt.colorbar()

    plt.subplot(2, 2, 4)
    plt.imshow(yn_pred_np, aspect="auto", cmap="inferno")
    plt.title("YN Prediction")
    plt.colorbar()

    plt.tight_layout()
    plt.show()

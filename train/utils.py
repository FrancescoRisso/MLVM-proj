from typing import Any

import matplotlib.pyplot as plt
import mido  # type: ignore
import numpy as np
import numpy.typing as npt
import torch
from matplotlib.axes import _axes  # type: ignore
from matplotlib.figure import Figure

from dataloader.Song import Song
from settings import Settings as s


def midi_to_label_matrices(
    mido_midi: mido.MidiFile, sample_rate: float, hop_length: int, n_bins: int = 88
):
    ticks_per_beat = mido_midi.ticks_per_beat
    min_pitch = 21  # Pitch corrispondente a "A0"
    max_pitch = min_pitch + n_bins

    # Tempo iniziale
    current_tempo = 500000  # Default 120 BPM
    time_in_seconds = 0.0
    tick_accumulator: int = 0

    # Lista delle note (pitch, start_time_sec, end_time_sec)
    active_notes: dict[int, float] = {}
    notes: list[tuple[int, float, torch.Tensor | float]] = []

    for msg in mido.merge_tracks(mido_midi.tracks):  # type: ignore
        assert isinstance(msg.time, int)  # type: ignore
        tick_accumulator += msg.time

        if msg.type == "set_tempo":  # type: ignore
            assert isinstance(msg.tempo, int)  # type: ignore
            current_tempo = msg.tempo  # type: ignore

        time_in_seconds = mido.tick2second(  # type: ignore
            tick_accumulator, ticks_per_beat, current_tempo
        )
        assert isinstance(time_in_seconds, torch.Tensor) or isinstance(
            time_in_seconds, float
        )

        if msg.type == "note_on" and msg.velocity > 0:  # type: ignore
            active_notes[msg.note] = time_in_seconds  # type: ignore

        elif (msg.type == "note_off") or (msg.type == "note_on" and msg.velocity == 0):  # type: ignore
            start = active_notes.pop(msg.note, None)  # type: ignore
            if start is not None and (min_pitch <= msg.note) and (msg.note < max_pitch):  # type: ignore
                assert isinstance(msg.note, int)  # type: ignore
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


def to_tensor(array: Any) -> torch.Tensor:
    return torch.tensor(array) if isinstance(array, np.ndarray) else array


def to_numpy(
    tensor: torch.Tensor | None | npt.NDArray[np.generic],
) -> npt.NDArray[np.generic] | None:
    if tensor is None:
        return None
    return tensor.detach().cpu().numpy() if isinstance(tensor, torch.Tensor) else tensor  # type: ignore


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
) -> dict[str, float]:
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


def imshow_fixed(
    ax: _axes.Axes, data: npt.NDArray[np.generic], title: str, fig: Figure
):
    data = np.squeeze(data)
    im = ax.imshow(data, aspect="auto", origin="lower", vmin=0, vmax=1)  # type: ignore
    ax.set_title(title)  # type: ignore
    fig.colorbar(im, ax=ax)  # type: ignore


def plot_harmoniccnn_outputs(
    yo: torch.Tensor, 
    yp: torch.Tensor, 
    yn: torch.Tensor | None, 
    title_prefix: str,
    ax=None  # Aggiunto parametro opzionale per l'asse
):
    yo_np = to_numpy(yo)
    yp_np = to_numpy(yp)
    yn_np = None if s.remove_yn else to_numpy(yn)

    assert yo_np is not None
    assert yp_np is not None

    # Se non viene passato un asse, crea una nuova figura
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure

    n_rows = 2 if s.remove_yn else 3
    axes = [ax] if n_rows == 1 else [ax] + [ax.twinx() for _ in range(n_rows-1)]

    imshow_fixed(axes[0], yo_np, f"YO {title_prefix}", fig)
    imshow_fixed(axes[1], yp_np, f"YP {title_prefix}", fig)

    if not s.remove_yn and yn_np is not None:
        imshow_fixed(axes[2], yn_np, f"YN {title_prefix}", fig)

    for ax in axes:
        ax.axis("off")

    return fig


def should_log_image(epoch: int) -> bool:
    if epoch < 10:
        return epoch % 2 == 0
    else:
        return epoch % 5 == 0


@torch.no_grad()  # type: ignore
def plot_fixed_sample(
    model: torch.nn.Module,
    sample: tuple[
        tuple[npt.NDArray[np.uint16], int, int, int],
        npt.NDArray[np.float32] | torch.Tensor,
    ],
    device: torch.device,
):
    (midi_np, tempo, ticks_per_beat, num_messages), audio = sample
    audio = torch.from_numpy(audio) if isinstance(audio, np.ndarray) else audio  # type: ignore
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

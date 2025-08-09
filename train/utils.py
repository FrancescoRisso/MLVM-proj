import os
from typing import Any

import librosa
import librosa.display
import matplotlib.pyplot as plt
import mido  # type: ignore
import numpy as np
import numpy.typing as npt
import torch
from matplotlib.axes import _axes  # type: ignore
from matplotlib.figure import Figure

from dataloader.Song import Song
from model.postprocessing import postprocess
from settings import Settings as s


def midi_to_label_matrices(
    mido_midi: mido.MidiFile, sample_rate: float, hop_length: int, n_bins: int = 88
):
    ticks_per_beat = mido_midi.ticks_per_beat if mido_midi.ticks_per_beat != 0 else 1
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
                notes.append((msg.note, start, time_in_seconds))  # type: ignore

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
    ax: _axes.Axes,
    data: npt.NDArray[np.generic],
    title: str,
    fig: Figure,
    add_colorbar: bool = True,
):
    data = np.squeeze(data)
    im = ax.imshow(data, aspect="auto", origin="lower", vmin=0, vmax=1)  # type: ignore
    ax.set_title(title)  # type: ignore

    if add_colorbar:
        fig.colorbar(im, ax=ax)  # type: ignore

    return


def plot_harmoniccnn_outputs(
    yo: torch.Tensor,
    yp: torch.Tensor,
    yn: torch.Tensor | None = None,
    title_prefix: str = "",
    add_colorbar: bool = True,
    vmin: float = 0.0,
    vmax: float = 1.0,
    cmap: str = "magma",
):
    n_plots = 2 + (1 if yn is not None else 0)
    fig, axs = plt.subplots(n_plots, 1, figsize=(6, 4 * n_plots))  # type: ignore

    ims: list[Any] = []

    im0 = axs[0].imshow(
        to_numpy(yo.squeeze()),
        aspect="auto",
        origin="lower",
        cmap=cmap,
        vmax=vmax,
        vmin=vmin,
    )
    axs[0].set_title(title_prefix + " yo")
    axs[0].axis("off")
    ims.append(im0)

    im1 = axs[1].imshow(
        to_numpy(yp.squeeze()),
        aspect="auto",
        origin="lower",
        cmap=cmap,
        vmax=vmax,
        vmin=vmin,
    )
    axs[1].set_title(title_prefix + " yp")
    axs[1].axis("off")
    ims.append(im1)

    if yn is not None:
        im2 = axs[2].imshow(
            to_numpy(yn.squeeze()),
            aspect="auto",
            origin="lower",
            cmap=cmap,
            vmax=vmax,
            vmin=vmin,
        )
        axs[2].set_title(title_prefix + " yn")
        axs[2].axis("off")
        ims.append(im2)

    if add_colorbar:
        for ax, im in zip(axs, ims):
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)  # type: ignore

    fig.tight_layout()
    return fig


def should_log_image(epoch: int) -> bool:
    if epoch < 10:
        return epoch % 2 == 0
    else:
        return epoch % 5 == 0


@torch.no_grad()  # type: ignore
def plot_fixed_sample(
    sample: tuple[
        tuple[npt.NDArray[np.uint16], int, int, int],
        npt.NDArray[np.float32] | torch.Tensor,
    ],
    device: torch.device,
    yo_pred: torch.Tensor,
    yp_pred: torch.Tensor,
    yn_pred: torch.Tensor | None,
):
    (midi_np, tempo, ticks_per_beat, num_messages), _ = sample

    midi = Song.from_np(midi_np, tempo, ticks_per_beat, num_messages).get_midi()
    yo_true, yp_true = midi_to_label_matrices(
        midi, s.sample_rate, s.hop_length, n_bins=88
    )
    yn_true = yp_true if s.remove_yn else yo_true

    yo_true = to_tensor(yo_true).unsqueeze(0).to(device)
    yp_true = to_tensor(yp_true).unsqueeze(0).to(device)
    yn_true = to_tensor(yn_true).unsqueeze(0).to(device) if not s.remove_yn else None

    yo_pred = torch.sigmoid(yo_pred).squeeze(1).cpu()
    yp_pred = torch.sigmoid(yp_pred).squeeze(1).cpu()
    yn_pred = (
        torch.sigmoid(yn_pred).squeeze(1).cpu()
        if (not s.remove_yn and yn_pred is not None)
        else None
    )

    title_prefix = "Prediction"
    fig = plot_harmoniccnn_outputs(yo_pred, yp_pred, yn_pred, title_prefix)

    # Postprocess to MIDI
    midi_out = postprocess(
        yo_pred,
        yp_pred,
        yn_pred if yn_pred is not None else yp_pred,
        s.seconds,
        s.sample_rate,
    )

    return fig, midi_out


def save_plot(
    sample: tuple[
        float,
        int,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
    ],
    name: str,
    output_dir: str,
):
    _, idx, audio_input, yo_pred, yp_pred, yn_pred, yo_true, yp_true, yn_true = sample

    yo_pred = (
        yo_pred.squeeze(0).squeeze(0) if len(yo_pred.shape) == 4 else yo_pred.squeeze(0)
    )
    yp_pred = (
        yp_pred.squeeze(0).squeeze(0) if len(yp_pred.shape) == 4 else yp_pred.squeeze(0)
    )
    yn_pred = (
        yn_pred.squeeze(0).squeeze(0)
        if yn_pred is not None and len(yn_pred.shape) == 4
        else yn_pred.squeeze(0) if yn_pred is not None else None
    )
    yo_true = (
        yo_true.squeeze(0).squeeze(0) if len(yo_true.shape) == 4 else yo_true.squeeze(0)
    )
    yp_true = (
        yp_true.squeeze(0).squeeze(0) if len(yp_true.shape) == 4 else yp_true.squeeze(0)
    )
    yn_true = (
        yn_true.squeeze(0).squeeze(0)
        if yn_true is not None and len(yn_true.shape) == 4
        else yn_true.squeeze(0) if yn_true is not None else None
    )

    audio_np = to_numpy(audio_input)
    audio_np = audio_np.squeeze() if audio_np is not None else None
    assert audio_np is not None

    # Calcola CQT
    cqt = librosa.cqt(  # type: ignore
        audio_np,
        sr=s.sample_rate,
        hop_length=s.hop_length,
        n_bins=88,
        bins_per_octave=12,
    )
    cqt_mag = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)  # type: ignore

    fig, axs = plt.subplots(3, 1, figsize=(12, 15))  # type: ignore

    # 1) Plot CQT
    img1 = librosa.display.specshow(  # type: ignore
        cqt_mag,
        sr=s.sample_rate,
        hop_length=s.hop_length,
        ax=axs[0],
    )
    axs[0].set_title(f"CQT - {name} sample idx {idx}")
    fig.colorbar(img1, ax=axs[0], format="%+2.0f dB")  # type: ignore

    # 2) Plot Yp_pred
    yp_pred_sig = to_numpy(torch.sigmoid(yp_pred))
    assert yp_pred_sig is not None

    im2 = axs[1].imshow(
        yp_pred_sig, aspect="auto", origin="lower", cmap="magma", vmin=0, vmax=1
    )
    axs[1].set_title(f"Prediction (Yp) - idx {idx}")
    axs[1].axis("off")
    fig.colorbar(im2, ax=axs[1])  # type: ignore

    # 3) Plot Yp_true
    yp_true_np = to_numpy(yp_true)
    assert yp_true is not None

    im3 = axs[2].imshow(
        yp_true_np, aspect="auto", origin="lower", cmap="magma", vmin=0, vmax=1
    )
    axs[2].set_title(f"Ground Truth (Yp) - idx {idx}")
    axs[2].axis("off")
    fig.colorbar(im3, ax=axs[2])  # type: ignore

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name}__{idx}.png"), dpi=150)  # type: ignore
    plt.close(fig)

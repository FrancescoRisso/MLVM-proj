from typing import Any

import matplotlib.pyplot as plt
import mido
import numpy as np
import numpy.typing as npt
import torch
from matplotlib.axes import _axes
from matplotlib.figure import Figure

import librosa
import librosa.display
import os
from tqdm.auto import tqdm

from torch.utils.data import DataLoader
from dataloader.dataset import DataSet
from dataloader.Song import Song
from dataloader.split import Split
from model.model import HarmonicCNN

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

    for msg in mido.merge_tracks(mido_midi.tracks):
        assert isinstance(msg.time, int)
        tick_accumulator += msg.time

        if msg.type == "set_tempo":
            assert isinstance(msg.tempo, int)
            current_tempo = msg.tempo

        time_in_seconds = mido.tick2second(
            tick_accumulator, ticks_per_beat, current_tempo
        )
        assert isinstance(time_in_seconds, torch.Tensor) or isinstance(
            time_in_seconds, float
        )

        if msg.type == "note_on" and msg.velocity > 0:
            active_notes[msg.note] = time_in_seconds

        elif (msg.type == "note_off") or (msg.type == "note_on" and msg.velocity == 0):
            start = active_notes.pop(msg.note, None)
            if start is not None and (min_pitch <= msg.note) and (msg.note < max_pitch):
                assert isinstance(msg.note, int)
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
    im = ax.imshow(data, aspect="auto", origin="lower", vmin=0, vmax=1)
    ax.set_title(title)

    if add_colorbar:
        fig.colorbar(im, ax=ax)

    return


def plot_harmoniccnn_outputs(
    yo,
    yp,
    yn=None,
    title_prefix="",
    add_colorbar=True,
    vmin=0.0,
    vmax=1.0,
    cmap="magma",
):
    n_plots = 2 + (1 if yn is not None else 0)
    fig, axs = plt.subplots(n_plots, 1, figsize=(6, 4 * n_plots))

    if n_plots == 1:
        axs = [axs]

    ims = []

    if yo is not None:
        im0 = axs[0].imshow(
            yo.squeeze().cpu().numpy(),
            aspect="auto",
            origin="lower",
            cmap=cmap,
            vmax=vmax,
            vmin=vmin,
        )
        axs[0].set_title(title_prefix + " yo")
        axs[0].axis("off")
        ims.append(im0)

    if yp is not None:
        im1 = axs[1].imshow(
            yp.squeeze().cpu().numpy(),
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
            yn.squeeze().cpu().numpy(),
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
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    return fig


def should_log_image(epoch: int) -> bool:
    if epoch < 10:
        return epoch % 2 == 0
    else:
        return epoch % 5 == 0


@torch.no_grad()
def plot_fixed_sample(
    model: torch.nn.Module,
    sample: tuple[
        tuple[npt.NDArray[np.uint16], int, int, int],
        npt.NDArray[np.float32] | torch.Tensor,
    ],
    device: torch.device,
):
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


def save_plot(sample, name, output_dir):
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

    audio_np = audio_input.numpy().squeeze()

    # Calcola CQT
    cqt = librosa.cqt(
        audio_np,
        sr=s.sample_rate,
        hop_length=s.hop_length,
        n_bins=88,
        bins_per_octave=12,
    )
    cqt_mag = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)

    fig, axs = plt.subplots(3, 1, figsize=(12, 15))

    # 1) Plot CQT
    img1 = librosa.display.specshow(
        cqt_mag,
        sr=s.sample_rate,
        hop_length=s.hop_length,
        ax=axs[0],
    )
    axs[0].set_title(f"CQT - {name} sample idx {idx}")
    fig.colorbar(img1, ax=axs[0], format="%+2.0f dB")

    # 2) Plot Yp_pred
    yp_pred_sig = torch.sigmoid(yp_pred).cpu().numpy()
    im2 = axs[1].imshow(
        yp_pred_sig, aspect="auto", origin="lower", cmap="magma", vmin=0, vmax=1
    )
    axs[1].set_title(f"Prediction (Yp) - idx {idx}")
    axs[1].axis("off")
    fig.colorbar(im2, ax=axs[1])

    # 3) Plot Yp_true
    yp_true_np = yp_true.cpu().numpy() if isinstance(yp_true, torch.Tensor) else yp_true
    im3 = axs[2].imshow(
        yp_true_np, aspect="auto", origin="lower", cmap="magma", vmin=0, vmax=1
    )
    axs[2].set_title(f"Ground Truth (Yp) - idx {idx}")
    axs[2].axis("off")
    fig.colorbar(im3, ax=axs[2])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name}__{idx}.png"), dpi=150)
    plt.close(fig)


def evaluate_and_plot_extremes(
    model_path: str, dataset: Split, output_dir: str = "eval_plots", top_k: int = 5
) -> None:
    device = s.device
    print(f"Evaluating model {model_path} for top/bottom F1 samples on {device}")

    model = HarmonicCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    val_dataset = DataSet(dataset, s.seconds)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    os.makedirs(output_dir, exist_ok=True)

    scores = []
    skipped_samples = 0

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(val_loader, desc="Computing F1 scores")):
            (midis_np, tempos, ticks_per_beats, nums_messages), audios = batch

            midi = Song.from_np(
                midis_np[0], tempos[0], ticks_per_beats[0], nums_messages[0]
            ).get_midi()
            yo_true, yp_true = midi_to_label_matrices(
                midi, s.sample_rate, s.hop_length, n_bins=88
            )
            yn_true = yp_true if not s.remove_yn else None

            yo_true_t = to_tensor(yo_true).to(device).unsqueeze(0)
            yp_true_t = to_tensor(yp_true).to(device).unsqueeze(0)
            yn_true_t = (
                to_tensor(yn_true).to(device).unsqueeze(0)
                if yn_true is not None
                else None
            )

            audio_input = audios.to(device)

            outputs = model(audio_input)
            if s.remove_yn:
                yo_pred = outputs[0]
                yp_pred = outputs[1]
                yn_pred = None
            else:
                yo_pred, yp_pred, yn_pred = outputs

            yp_pred_sig = torch.sigmoid(yp_pred).squeeze(1)

            gt_all_zeros = torch.all(yp_true_t == 0).item()
            pred_thresholded = (yp_pred_sig >= 0.5).float()
            pred_all_zeros = torch.all(pred_thresholded == 0).item()

            if gt_all_zeros and pred_all_zeros:
                skipped_samples += 1
                continue

            metrics = binary_classification_metrics(yp_pred_sig, yp_true_t)
            tp, fp, fn = metrics["TP"], metrics["FP"], metrics["FN"]
            f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)

            scores.append(
                (
                    f1,
                    idx,
                    audio_input.cpu(),
                    yo_pred.cpu(),
                    yp_pred.cpu(),
                    yn_pred.cpu() if yn_pred is not None else None,
                    yo_true_t.cpu(),
                    yp_true_t.cpu(),
                    yn_true_t.cpu() if yn_true_t is not None else None,
                )
            )

    scores_sorted = sorted(scores, key=lambda x: x[0], reverse=True)
    top_samples = scores_sorted[:top_k]
    bottom_samples = scores_sorted[-top_k:]

    for sample in top_samples:
        save_plot(sample, "best", output_dir)
    for sample in bottom_samples:
        save_plot(sample, "worst", output_dir)

    print(f"Saved {top_k} best and {top_k} worst plots to '{output_dir}'")
    print(
        f"Skipped {skipped_samples} samples where both ground truth and prediction were all zeros"
    )

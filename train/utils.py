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


def weighted_soft_accuracy(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    weight_positive: float = 0.95,
    weight_negative: float = 0.05,
    tolerance: float = 0.1,
) -> float:
    """
    Accuratezza pesata: le predizioni corrette per classi positive (1) valgono di più (0.95),
    mentre quelle per classi negative (0) valgono meno (0.05).
    E' intesa per essere usata come debug per vedere se la rete sta effettivamente migliorando
    Non è intesa per essere usata come metrica di valutazione finale.
    """
    with torch.no_grad():
        diff = torch.abs(y_pred - y_true)
        correct = (diff <= tolerance).float()

        # Assegna pesi: 0.95 per le note attive, 0.05 per quelle inattive
        weights = torch.where(y_true > 0.5, weight_positive, weight_negative)
        weighted_correct = correct * weights

        return weighted_correct.sum().item() / weights.sum().item()


def plot_prediction_vs_ground_truth(yo_pred, yp_pred, yn_pred, yo_true, yp_true, yn_true):
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
    im0 = axes[0, 0].imshow(yo_true_np, aspect='auto', origin='lower')
    axes[0, 0].set_title("YO Ground Truth")
    fig.colorbar(im0, ax=axes[0, 0])
    
    im1 = axes[0, 1].imshow(yo_pred_np, aspect='auto', origin='lower')
    axes[0, 1].set_title("YO Prediction")
    fig.colorbar(im1, ax=axes[0, 1])

    # Plot YP
    im2 = axes[1, 0].imshow(yp_true_np, aspect='auto', origin='lower')
    axes[1, 0].set_title("YP Ground Truth")
    fig.colorbar(im2, ax=axes[1, 0])
    
    im3 = axes[1, 1].imshow(yp_pred_np, aspect='auto', origin='lower')
    axes[1, 1].set_title("YP Prediction")
    fig.colorbar(im3, ax=axes[1, 1])

    # Plot YN (solo se non rimosso)
    if not s.remove_yn:
        im4 = axes[2, 0].imshow(yn_true_np, aspect='auto', origin='lower')
        axes[2, 0].set_title("YN Ground Truth")
        fig.colorbar(im4, ax=axes[2, 0])
        
        im5 = axes[2, 1].imshow(yn_pred_np, aspect='auto', origin='lower')
        axes[2, 1].set_title("YN Prediction")
        fig.colorbar(im5, ax=axes[2, 1])

    for ax_row in axes:
        for ax in ax_row:
            ax.axis('off')

    plt.tight_layout()
    return fig


def should_log_image(epoch):
    # Logga ogni 2 epoche per le prime 10, poi ogni 5 epoche
    if epoch <= 10:
        return epoch % 2 == 0
    else:
        return epoch % 5 == 0


def batch_prediction_plot():

    import os
    from dataloader.dataset import DataSet, Split
    from model.model import HarmonicCNN
    from torch.utils.data import DataLoader
    from settings import Settings as s

    device = s.device
    print(f"Eval on {device}")

    model_path = os.path.join(
        "model_saves", "training_2025-06-25_10-19-45", "harmoniccnn_epoch_5.pth"
    )

    run_single_batch_prediction_plot(
        model_path=model_path,
        model=HarmonicCNN().to(s.device),
        dataloader=DataLoader(
            DataSet(Split.TRAIN, s.seconds), batch_size=s.batch_size, shuffle=True
        ),
    )


@torch.no_grad()
def run_single_batch_prediction_plot(
    model_path: str,
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # --- Carica un batch ---
    batch = next(iter(dataloader))
    (midis_np, tempos, ticks_per_beats, nums_messages), audios = batch

    # --- Crea ground truth ---
    midi = Song.from_np(
        midis_np[0], tempos[0], ticks_per_beats[0], nums_messages[0]
    ).get_midi()
    yo_true, yn_true = midi_to_label_matrices(
        midi, s.sample_rate, s.hop_length, n_bins=88
    )

    # --- Input e target su device ---
    input_audio = audios[0].to(device).unsqueeze(0)
    yo_true_tensor = to_tensor(yo_true).to(device).squeeze(0)
    yn_true_tensor = to_tensor(yn_true).to(device).squeeze(0)

    # --- Predizione ---
    yo_logits, yp_logits, yn_logits = model(input_audio)
    yo_logits = yo_logits.squeeze(1)[0]
    yp_logits = yp_logits.squeeze(1)[0]
    if yn_logits is not None:
        yn_logits = yn_logits.squeeze(1)[0]

    # --- Applica sigmoid per accuratezze soft ---
    yo_pred = torch.sigmoid(yo_logits)
    yp_pred = torch.sigmoid(yp_logits) 
    if yn_logits is not None:
        yn_pred = torch.sigmoid(yn_logits)

    # --- Calcolo loss totale ---
    loss_total = harmoniccnn_loss(
        yo_logits=yo_logits,
        yp_logits=yp_logits,
        yo_true=yo_true_tensor,
        yp_true=yn_true_tensor,
        yn_logits=yn_logits,
        yn_true=yn_true_tensor,
        label_smoothing=0.2,
        weighted=False,
        positive_weight=0.5,
    )
    total_loss = sum(loss_total.values())
    # --- Calcolo accuratezze ---
    acc_yo = weighted_soft_accuracy(yo_pred, yo_true_tensor)
    acc_yp = weighted_soft_accuracy(yp_pred, yn_true_tensor)
    if yn_logits is not None:
        acc_yn = weighted_soft_accuracy(yn_pred, yn_true_tensor)

    # --- Stampa risultati ---
    print(f"Total Loss: {total_loss.item():.4f}")
    print(f"YO Soft Accuracy: {acc_yo:.4f}")
    print(f"YN Soft Accuracy: {acc_yn:.4f}")

    # --- Plot ---
    plot_prediction_vs_ground_truth(yo_pred, yp_pred, yo_true_tensor, yn_true_tensor)

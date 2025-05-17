import torch
import torch.nn.functional as F
from typing import Callable, Dict


def apply_label_smoothing(y_true: torch.Tensor, smoothing: float) -> torch.Tensor:
    """Applica label smoothing alle etichette binarie."""
    return y_true * (1 - smoothing) + 0.5 * smoothing


def transcription_loss(
    y_true: torch.Tensor, y_logits: torch.Tensor, label_smoothing: float
) -> torch.Tensor:
    """Binary cross entropy con label smoothing (non pesata)."""
    y_true = apply_label_smoothing(y_true, label_smoothing)
    return F.binary_cross_entropy_with_logits(y_logits, y_true)


def weighted_transcription_loss(
    y_true: torch.Tensor,
    y_logits: torch.Tensor,
    label_smoothing: float,
    positive_weight: float = 0.5,
) -> torch.Tensor:
    """Binary cross entropy con pesi diversi per classi positive e negative."""
    y_true = apply_label_smoothing(y_true, label_smoothing)

    # Maschere per positivi e negativi
    negative_mask = y_true == 0
    positive_mask = y_true != 0

    bce_neg = F.binary_cross_entropy_with_logits(
        y_logits[negative_mask], y_true[negative_mask], reduction="sum"
    ) / (negative_mask.sum().float().clamp(min=1))

    bce_pos = F.binary_cross_entropy_with_logits(
        y_logits[positive_mask], y_true[positive_mask], reduction="sum"
    ) / (positive_mask.sum().float().clamp(min=1))

    return (1 - positive_weight) * bce_neg + positive_weight * bce_pos


def onset_loss(
    weighted: bool, label_smoothing: float, positive_weight: float
) -> Callable:
    """Restituisce una funzione di loss per l'onset (pesata o no)."""
    if weighted:
        return lambda y_true, y_logits: weighted_transcription_loss(
            y_true, y_logits, label_smoothing, positive_weight
        )
    return lambda y_true, y_logits: transcription_loss(
        y_true, y_logits, label_smoothing
    )


def loss(
    label_smoothing: float = 0.2, weighted: bool = False, positive_weight: float = 0.5
) -> Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]:
    """Restituisce le funzioni di loss per 'note' e 'onset'."""
    return {
        "note": lambda y_true, y_logits: transcription_loss(
            y_true, y_logits, label_smoothing
        ),
        "onset": onset_loss(weighted, label_smoothing, positive_weight),
    }


def harmoniccnn_loss(
    yo_logits: torch.Tensor,
    yn_logits: torch.Tensor,
    yo_true: torch.Tensor,
    yn_true: torch.Tensor,
    label_smoothing: float = 0.2,
    weighted: bool = False,
    positive_weight: float = 0.5,
) -> torch.Tensor:
    """Loss finale per HarmonicCNN, combinando onset e note."""
    losses = loss(label_smoothing, weighted, positive_weight)

    loss_onset = losses["onset"](yo_true, yo_logits)
    loss_note = losses["note"](yn_true, yn_logits)

    return loss_onset + loss_note

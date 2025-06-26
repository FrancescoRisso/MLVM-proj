import torch
import torch.nn.functional as F
from typing import Callable, Dict


def apply_label_smoothing(y_true: torch.Tensor, smoothing: float) -> torch.Tensor:
    """Applies label smoothing to binary labels."""
    return y_true * (1 - smoothing) + 0.5 * smoothing


def transcription_loss(
    y_true: torch.Tensor, y_logits: torch.Tensor, label_smoothing: float
) -> torch.Tensor:
    """Binary cross entropy with label smoothing (not weighted)."""
    y_true = apply_label_smoothing(y_true, label_smoothing)
    return F.binary_cross_entropy_with_logits(y_logits, y_true)


def weighted_transcription_loss(
    y_true: torch.Tensor,
    y_logits: torch.Tensor,
    label_smoothing: float,
    positive_weight: float = 0.5,
) -> torch.Tensor:
    """Binary cross entropy with different weights for positive and negative classes."""
    y_true = apply_label_smoothing(y_true, label_smoothing)

    # Masks for positives and negatives
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
    """Returns a loss function for onset (weighted or not)."""
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
    """Returns the loss functions for 'note' and 'onset'."""
    return {
        "note": lambda y_true, y_logits: transcription_loss(
            y_true, y_logits, label_smoothing
        ),
        "onset": onset_loss(weighted, label_smoothing, positive_weight),
    }

def harmoniccnn_loss(
    yo_logits: torch.Tensor,
    yp_logits: torch.Tensor,
    yo_true: torch.Tensor,
    yp_true: torch.Tensor,
    yn_logits: torch.Tensor = None,
    yn_true: torch.Tensor = None,
    label_smoothing: float = 0.2,
    weighted: bool = False,
    positive_weight: float = 0.5,
) -> torch.Tensor:
    """Final loss for HarmonicCNN, combining onset and notes."""
    losses = loss(label_smoothing, weighted, positive_weight)

    loss_onset = losses["onset"](yo_true, yo_logits)
    loss_tone = losses["note"](yp_true, yp_logits)
    if yn_logits is None:
        return {
            'loss_yo': loss_onset,
            'loss_yp': loss_tone
        }

    loss_note = losses["note"](yn_true, yn_logits)

    return {
        'loss_yo': loss_onset,
        'loss_yp': loss_tone,
        'loss_yn': loss_note
    }

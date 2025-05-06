import torch
import torch.nn.functional as F
from typing import Callable, Dict, Any

def safe_bce(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Safely compute BCE avoiding empty tensors and log(0)."""
    if pred.numel() == 0:
        return torch.tensor(0.0, device=pred.device)
    pred = torch.clamp(pred, 1e-7, 1 - 1e-7)
    return F.binary_cross_entropy(pred, target)


def transcription_loss(y_true: torch.Tensor, y_pred: torch.Tensor, label_smoothing: float) -> torch.Tensor:
    """Binary cross entropy loss with optional label smoothing."""
    y_pred = torch.clamp(y_pred, 1e-7, 1 - 1e-7)
    y_true = y_true * (1 - label_smoothing) + 0.5 * label_smoothing
    return F.binary_cross_entropy(y_pred, y_true)


def weighted_transcription_loss(y_true: torch.Tensor, y_pred: torch.Tensor, label_smoothing: float, positive_weight: float = 0.5) -> torch.Tensor:
    """Weighted binary cross entropy loss."""
    y_true = y_true * (1 - label_smoothing) + 0.5 * label_smoothing

    negative_mask = (y_true == 0)
    nonnegative_mask = ~negative_mask

    bce_negative = safe_bce(y_pred[negative_mask], y_true[negative_mask])
    bce_nonnegative = safe_bce(y_pred[nonnegative_mask], y_true[nonnegative_mask])

    return (1 - positive_weight) * bce_negative + positive_weight * bce_nonnegative


def onset_loss(weighted: bool, label_smoothing: float, positive_weight: float) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    if weighted:
        return lambda y_true, y_pred: weighted_transcription_loss(y_true, y_pred, label_smoothing, positive_weight)
    return lambda y_true, y_pred: transcription_loss(y_true, y_pred, label_smoothing)


def loss(label_smoothing: float = 0.2, weighted: bool = False, positive_weight: float = 0.5) -> Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]:
    """Returns a dictionary of loss functions."""
    return {
        "note": lambda y_true, y_pred: transcription_loss(y_true, y_pred, label_smoothing),
        "onset": onset_loss(weighted, label_smoothing, positive_weight)
    }


def harmoniccnn_loss(
    yo_pred: torch.Tensor,
    yn_pred: torch.Tensor,
    yo_true: torch.Tensor,
    yn_true: torch.Tensor,
    label_smoothing: float = 0.2,
    weighted: bool = False,
    positive_weight: float = 0.5
) -> torch.Tensor:
    
    """Combines onset and note losses for HarmonicCNN."""
    
    losses = loss(label_smoothing, weighted, positive_weight)
    
    loss_onset = losses["onset"](yo_true, yo_pred)
    loss_note = losses["note"](yn_true, yn_pred)
    
    return loss_onset + loss_note

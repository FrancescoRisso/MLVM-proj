import torch
import torch.nn.functional as F
from typing import Callable
from settings import Settings as s


def apply_label_smoothing(y_true: torch.Tensor, smoothing: float) -> torch.Tensor:
    """Applica label smoothing a etichette binarie."""
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
    positive_weight: float,
) -> torch.Tensor:
    """Binary cross entropy con pesi diversi per classi positive e negative."""
    y_true = apply_label_smoothing(y_true, label_smoothing)

    # Mask per negativi e positivi
    negative_mask = y_true == 0
    positive_mask = y_true != 0

    bce_neg = F.binary_cross_entropy_with_logits(
        y_logits[negative_mask], y_true[negative_mask], reduction="sum"
    ) / negative_mask.sum().float().clamp(min=1)

    bce_pos = F.binary_cross_entropy_with_logits(
        y_logits[positive_mask], y_true[positive_mask], reduction="sum"
    ) / positive_mask.sum().float().clamp(min=1)

    return (1 - positive_weight) * bce_neg + positive_weight * bce_pos


def harmoniccnn_loss(
    yo_logits: torch.Tensor,
    yp_logits: torch.Tensor,
    yo_true: torch.Tensor,
    yp_true: torch.Tensor,
    yn_logits: torch.Tensor = None,
    yn_true: torch.Tensor = None,
    label_smoothing: float = 0.0,
    weighted: bool = False,
) -> dict:
    """
    Loss finale per HarmonicCNN, con smoothing e pesi personalizzabili per onset, tone, note.
    Usa sempre weighted_transcription_loss.
    """
    # Se weighted=False, forziamo pesi bilanciati
    if not weighted:
        positive_weight_yo = 0.5
        positive_weight_yp = 0.5
        positive_weight_yn = 0.5
    else:
        positive_weight_yo = s.positive_weight_yo
        positive_weight_yp = s.positive_weight_yp
        positive_weight_yn = s.positive_weight_yn

    loss_onset = weighted_transcription_loss(
        yo_true, yo_logits, label_smoothing, positive_weight_yo
    )

    loss_tone = weighted_transcription_loss(
        yp_true, yp_logits, label_smoothing, positive_weight_yp
    )

    results = {"loss_yo": loss_onset, "loss_yp": loss_tone}

    if yn_logits is not None and yn_true is not None:
        loss_note = weighted_transcription_loss(
            yn_true, yn_logits, label_smoothing, positive_weight_yn
        )
        results["loss_yn"] = loss_note

    return results

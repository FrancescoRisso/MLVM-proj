import torch
import torch.nn.functional as F
from typing import Callable, Dict, Any

def transcription_loss(y_true: torch.Tensor, y_pred: torch.Tensor, label_smoothing: float) -> torch.Tensor:
    """Binary cross entropy loss between predicted posteriorgrams and ground truth matrices.
    
    Args:
        y_true: The true labels.
        y_pred: The predicted labels.
        label_smoothing: Squeeze labels towards 0.5.
    
    Returns:
        The transcription loss.
    """
    y_true = y_true * (1 - label_smoothing) + 0.5 * label_smoothing
    return F.binary_cross_entropy(y_pred, y_true)


def weighted_transcription_loss(y_true: torch.Tensor, y_pred: torch.Tensor, label_smoothing: float, positive_weight: float = 0.5) -> torch.Tensor:
    """Binary cross entropy loss with a weighting factor for positive and negative labels.
    
    Args:
        y_true: The true labels.
        y_pred: The predicted labels.
        label_smoothing: Smoothing factor. Squeezes labels towards 0.5.
        positive_weight: Weighting factor for the positive labels.
    
    Returns:
        The weighted transcription loss.
    """
    y_true = y_true * (1 - label_smoothing) + 0.5 * label_smoothing
    
    negative_mask = (y_true == 0)
    nonnegative_mask = ~negative_mask
    
    bce_negative = F.binary_cross_entropy(y_pred[negative_mask], y_true[negative_mask])
    bce_nonnegative = F.binary_cross_entropy(y_pred[nonnegative_mask], y_true[nonnegative_mask])
    
    return (1 - positive_weight) * bce_negative + positive_weight * bce_nonnegative


def onset_loss(weighted: bool, label_smoothing: float, positive_weight: float) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Returns a function that calculates the transcription loss, optionally weighted.
    
    Args:
        weighted: Whether or not to use a weighted cross entropy loss.
        label_smoothing: Smoothing factor. Squeezes labels towards 0.5.
        positive_weight: Weighting factor for the positive labels.
    
    Returns:
        A function for calculating the onset loss.
    """
    if weighted:
        return lambda x, y: weighted_transcription_loss(x, y, label_smoothing=label_smoothing, positive_weight=positive_weight)
    return lambda x, y: transcription_loss(x, y, label_smoothing=label_smoothing)


def loss(label_smoothing: float = 0.2, weighted: bool = False, positive_weight: float = 0.5) -> Dict[str, Any]:
    """Creates a dictionary of loss functions for contour, note, and onset posteriorgrams.
    
    Args:
        label_smoothing: Smoothing factor. Squeezes labels towards 0.5.
        weighted: Whether or not to use a weighted cross entropy loss.
        positive_weight: Weighting factor for the positive labels.
    
    Returns:
        A dictionary with keys "contour," "note," and "onset" containing loss functions.
    """
    loss_fn = lambda x, y: transcription_loss(x, y, label_smoothing=label_smoothing)
    loss_onset = onset_loss(weighted, label_smoothing, positive_weight)
    return {
        "contour": loss_fn,
        "note": loss_fn,
        "onset": loss_onset,
    }


def harmoniccnn_loss(
    yo_pred: torch.Tensor,
    yp_pred: torch.Tensor, 
    yn_pred: torch.Tensor,
    yo_true: torch.Tensor,
    yp_true: torch.Tensor,
    yn_true: torch.Tensor,
    label_smoothing: float = 0.2,
    weighted: bool = False,
    positive_weight: float = 0.5
) -> torch.Tensor:
    """
    Xalculates the total loss for the HarmonicCNN model, combining the losses for:
    onset (yo), contour (yp), e note (yn).
    """
    losses = loss(label_smoothing=label_smoothing, weighted=weighted, positive_weight=positive_weight)
    
    loss_onset = losses["onset"](yo_true, yo_pred)
    loss_contour = losses["contour"](yp_true, yp_pred)
    loss_note = losses["note"](yn_true, yn_pred)
    
    total_loss = loss_onset + loss_contour + loss_note
    return total_loss

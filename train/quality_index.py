import torch
from settings import Settings as s

def evaluate_note_prediction(
    yo_gt: torch.Tensor,
    yp_gt: torch.Tensor,
    yn_gt: torch.Tensor | None,
    yo_pred: torch.Tensor,
    yp_pred: torch.Tensor,
    yn_pred: torch.Tensor | None,
    onset_tol: float = 0.05,         # seconds (50 ms)
    pitch_tol: float = 0.25,         # in semitones (quarter tone = 0.25) — valid if pitch in MIDI
):
    
    if yn_gt is None:
        yn_gt = yp_gt

    if yn_pred is None:
        yn_pred = yp_pred

    onsets_correct = yo_gt > s.threshold
    onsets_predicted = yo_pred > s.threshold
    pitches_correct = yp_gt > s.threshold
    pitches_predicted = yp_pred > s.threshold
    notes_correct = yn_gt > s.threshold
    notes_predicted = yn_pred > s.threshold

    num_pitches, num_frames = yo_gt.shape

    for t in range(num_frames):
        for pitch in range(num_pitches):
            # TODO
            print()
    return

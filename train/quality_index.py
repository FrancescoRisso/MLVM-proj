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
) -> dict:
    
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

    # Time resolution (seconds per frame)
    time_per_frame = s.hop_length / s.sr
    onset_tol_frames = int(onset_tol / time_per_frame)

    matched_gt = torch.zeros_like(yn_gt, dtype=torch.bool)
    matched_pred = torch.zeros_like(yn_pred, dtype=torch.bool)

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # Get predicted notes: (pitch, onset_frame, offset_frame)
    pred_notes = []
    for pitch in range(num_pitches):
        active = False
        onset = 0
        for t in range(num_frames):
            if notes_predicted[pitch, t] and not active:
                onset = t
                active = True
            elif not notes_predicted[pitch, t] and active:
                offset = t
                active = False
                pred_notes.append((pitch, onset, offset))
        if active:
            pred_notes.append((pitch, onset, num_frames))

    # Get ground truth notes: (pitch, onset_frame, offset_frame)
    gt_notes = []
    for pitch in range(num_pitches):
        active = False
        onset = 0
        for t in range(num_frames):
            if notes_correct[pitch, t] and not active:
                onset = t
                active = True
            elif not notes_correct[pitch, t] and active:
                offset = t
                active = False
                gt_notes.append((pitch, onset, offset))
        if active:
            gt_notes.append((pitch, onset, num_frames))

    # Match predicted notes with ground truth notes
    matched_gt_flags = [False] * len(gt_notes)

    for pred_pitch, pred_onset, pred_offset in pred_notes:
        matched = False
        pred_duration = pred_offset - pred_onset

        for i, (gt_pitch, gt_onset, gt_offset) in enumerate(gt_notes):
            if matched_gt_flags[i]:
                continue

            gt_duration = gt_offset - gt_onset

            # Check pitch match
            if abs(pred_pitch - gt_pitch) > pitch_tol:
                continue

            # Check onset match
            if abs(pred_onset - gt_onset) > onset_tol_frames:
                continue

            # Check duration match
            if abs(pred_duration - gt_duration) > 0.2 * gt_duration:
                continue

            matched = True
            matched_gt_flags[i] = True
            true_positives += 1
            break

        if not matched:
            false_positives += 1

    false_negatives = len(gt_notes) - sum(matched_gt_flags)

    # Optionally return metrics
    precision = true_positives / (true_positives + false_positives + 1e-9)
    recall = true_positives / (true_positives + false_negatives + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }

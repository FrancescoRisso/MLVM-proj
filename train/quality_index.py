import torch

from settings import Settings as s


def check_note_quality(
    notes_tensor: torch.Tensor,
    num_pitches: int,
    num_frames: int,
) -> list[tuple[int, int, int]]:

    notes: list[tuple[int, int, int]] = []
    for pitch in range(num_pitches):
        active = False
        onset = 0
        for t in range(num_frames):
            if notes_tensor[pitch, t] and not active:
                onset = t
                active = True
            elif not notes_tensor[pitch, t] and active:
                offset = t
                active = False
                notes.append((pitch, onset, offset))
        if active:
            notes.append((pitch, onset, num_frames))

    return notes


def evaluate_note_prediction(
    yo_gt: torch.Tensor,
    yp_gt: torch.Tensor,
    yn_gt: torch.Tensor | None,
    yo_pred: torch.Tensor,
    yp_pred: torch.Tensor,
    yn_pred: torch.Tensor | None,
    onset_tol: float = 0.05,  # seconds (50 ms)
    note_tol: float = 0.2,  # 20%
    debug: bool = False,
) -> dict[str, float]:

    yo_gt = torch.sigmoid(yo_gt).squeeze(1).squeeze(0)
    yp_gt = torch.sigmoid(yp_gt).squeeze(1).squeeze(0)
    yn_gt = (
        torch.sigmoid(yn_gt).squeeze(1).squeeze(0)
        if (s.remove_yn == False and yn_gt is not None)
        else yp_gt
    )

    yo_pred = torch.sigmoid(yo_pred).squeeze(1).squeeze(0)
    yp_pred = torch.sigmoid(yp_pred).squeeze(1).squeeze(0)
    yn_pred = (
        torch.sigmoid(yn_pred).squeeze(1).squeeze(0)
        if (s.remove_yn == False and yn_pred is not None)
        else yp_pred
    )

    notes_correct = yn_gt > s.threshold
    notes_predicted = yn_pred > s.threshold
    num_pitches, num_frames = yo_gt.shape

    # Time resolution (seconds per frame)
    time_per_frame = s.hop_length / s.sample_rate
    onset_tol_frames = int(onset_tol / time_per_frame)

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # --- NOTE LEVEL ---
    pred_notes = check_note_quality(notes_predicted, num_pitches, num_frames)
    gt_notes = check_note_quality(notes_correct, num_pitches, num_frames)
    matched_gt_flags = [False] * len(gt_notes)

    for pred_pitch, pred_onset, pred_offset in pred_notes:
        matched = False
        pred_duration = pred_offset - pred_onset

        for i, (gt_pitch, gt_onset, gt_offset) in enumerate(gt_notes):
            if matched_gt_flags[i]:
                continue

            gt_duration = gt_offset - gt_onset

            if pred_pitch != gt_pitch:
                continue

            if abs(pred_onset - gt_onset) > onset_tol_frames:
                continue

            if abs(pred_duration - gt_duration) > note_tol * gt_duration:
                continue

            matched = True
            matched_gt_flags[i] = True
            true_positives += 1
            break

        if not matched:
            false_positives += 1

    false_negatives = len(gt_notes) - sum(matched_gt_flags)

    # --- BIN LEVEL ---
    TP_bins = torch.sum(notes_predicted & notes_correct).item()
    FP_bins = torch.sum(notes_predicted & ~notes_correct).item()
    FN_bins = torch.sum(~notes_predicted & notes_correct).item()

    # --- METRICS ---
    precision = true_positives / (true_positives + false_positives + 1e-9)
    recall = true_positives / (true_positives + false_negatives + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    precision_bins = TP_bins / (TP_bins + FP_bins + 1e-9)
    recall_bins = TP_bins / (TP_bins + FN_bins + 1e-9)
    f1_bins = 2 * precision_bins * recall_bins / (precision_bins + recall_bins + 1e-9)

    if debug:
        print(f"NOTE-LEVEL ---")
        print(f"True Positives: {true_positives}")
        print(f"False Positives: {false_positives}")
        print(f"False Negatives: {false_negatives}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"BIN-LEVEL ---")
        print(f"TP_bins: {TP_bins}, FP_bins: {FP_bins}, FN_bins: {FN_bins}")
        print(f"Precision_bins: {precision_bins:.4f}")
        print(f"Recall_bins: {recall_bins:.4f}")
        print(f"F1_bins: {f1_bins:.4f}")

    return {
        "TP": true_positives,
        "FP": false_positives,
        "FN": false_negatives,
        "TP_bins": TP_bins,
        "FP_bins": FP_bins,
        "FN_bins": FN_bins,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }

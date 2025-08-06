
import numpy as np
from typing import Tuple, Dict, List

def evaluate_note_prediction(
    gt_onsets: np.ndarray,
    gt_pitches: np.ndarray,
    gt_offsets: np.ndarray,
    pred_onsets: np.ndarray,
    pred_pitches: np.ndarray,
    pred_offsets: np.ndarray,
    onset_tol: float = 0.05,         # seconds (50 ms)
    pitch_tol: float = 0.25,         # in semitones (quarter tone = 0.25) — valid if pitch in MIDI
    pitch_unit: str = "midi"         # "midi", "hz", or "cents"
) -> Dict:
    """
    Evaluates the quality of note prediction.
    
    A predicted note is considered correct (true positive) if:
      - |pred_onset - gt_onset| <= onset_tol
      - pitch difference <= pitch_tol (in the system specified by pitch_unit)
      - |pred_offset - gt_offset| <= 0.2 * gt_duration  (gt_duration = gt_offset - gt_onset)
    
    Matching method: greedy based on onset (each prediction tries to match to the closest GT in onset
    among those not yet matched and with onset deviation <= onset_tol).
    
    Returns a dict containing TP, FP, FN, precision, recall, f1, accuracy, and the matched pairs.
    """
    # Basic validations
    gt_onsets = np.asarray(gt_onsets).astype(float)
    gt_pitches = np.asarray(gt_pitches).astype(float)
    gt_offsets = np.asarray(gt_offsets).astype(float)
    pred_onsets = np.asarray(pred_onsets).astype(float)
    pred_pitches = np.asarray(pred_pitches).astype(float)
    pred_offsets = np.asarray(pred_offsets).astype(float)
    
    assert gt_onsets.shape == gt_pitches.shape == gt_offsets.shape, "GT arrays must have the same shape"
    assert pred_onsets.shape == pred_pitches.shape == pred_offsets.shape, "Pred arrays must have the same shape"
    
    # Function to convert pitch -> semitones (reference: MIDI)
    def pitch_diff_in_semitones(p1, p2):
        if pitch_unit == "midi":
            return np.abs(p1 - p2)
        elif pitch_unit == "cents":
            # p in cents with respect to some reference: 100 cents = 1 semitone
            return np.abs(p1 - p2) / 100.0
        elif pitch_unit == "hz":
            # approximate conversion using log2: semitones = 12 * log2(f2/f1)
            # note: if using Hz, values must be positive and non-null
            # here we return |semitones| between two frequencies
            # for convenience, handle array/scalars
            p1 = np.asarray(p1, dtype=float)
            p2 = np.asarray(p2, dtype=float)
            # avoid division by zero
            eps = 1e-9
            ratio = (p1 + eps) / (p2 + eps)
            return np.abs(12.0 * np.log2(ratio))
        else:
            raise ValueError("pitch_unit must be 'midi', 'cents' or 'hz'")
    
    n_gt = len(gt_onsets)
    n_pred = len(pred_onsets)
    
    # Sort predictions by onset for deterministic behavior
    pred_order = np.argsort(pred_onsets)
    gt_order = np.argsort(gt_onsets)  # useful if we want to search faster
    
    matched_gt = np.full(n_gt, False)   # flag for GT already matched
    matched_pred = np.full(n_pred, False)
    matches: List[Tuple[int,int]] = []  # (idx_pred, idx_gt) obtained pairs (original index)
    
    # For each prediction (in onset order) look for the unmatched GT with onset diff <= onset_tol
    for p_idx in pred_order:
        p_on = pred_onsets[p_idx]
        # look for GT candidates within onset tolerance and not yet matched
        onset_diffs = np.abs(gt_onsets - p_on)
        candidate_idxs = np.where((onset_diffs <= onset_tol) & (~matched_gt))[0]
        if candidate_idxs.size == 0:
            continue  # no candidate for this prediction
        # among candidates, choose the GT with the closest onset (min onset diff)
        best_rel = candidate_idxs[np.argmin(onset_diffs[candidate_idxs])]
        # now check pitch and offset against that GT
        pitch_diff = pitch_diff_in_semitones(pred_pitches[p_idx], gt_pitches[best_rel])
        gt_duration = max(1e-9, gt_offsets[best_rel] - gt_onsets[best_rel])  # avoid zero
        offset_diff = np.abs(pred_offsets[p_idx] - gt_offsets[best_rel])
        # correctness criteria
        if (pitch_diff <= pitch_tol) and (offset_diff <= 0.2 * gt_duration):
            # consider this as TP (valid matching)
            matched_gt[best_rel] = True
            matched_pred[p_idx] = True
            matches.append((p_idx, best_rel))
        else:
            # does not satisfy pitch/offset, do not match; but we may not want to match to other GTs
            # decision: do NOT mark matched_gt, so GT remains available for other predictions
            # but to avoid multiple useless attempts we could optionally block; here we do not block.
            pass
    
    TP = len(matches)
    FP = np.sum(~matched_pred)
    FN = np.sum(~matched_gt)
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = TP / n_gt if n_gt > 0 else 0.0
    
    return {
        "TP": int(TP),
        "FP": int(FP),
        "FN": int(FN),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),      # TP / number of GT notes
        "matches": matches,               # list of tuples (idx_pred, idx_gt)
        "matched_pred_mask": matched_pred,
        "matched_gt_mask": matched_gt
    }

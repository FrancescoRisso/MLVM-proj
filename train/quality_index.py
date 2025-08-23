import torch
from settings import Settings as s
import numpy as np

def extract_notes_from_tensor(tensor: torch.Tensor) -> list[tuple[int, int, int]]:
    """
    Estrae note da un tensore binario (pitch x time)
    Restituisce lista di tuple (pitch, onset_frame, offset_frame)
    """
    if tensor.dim() == 3:
        tensor = tensor.squeeze(0)  # da [1, 88, 87] a [88, 87]

    notes = []
    num_pitches, num_frames = tensor.shape
    threshold = s.threshold  # soglia per considerare attiva una nota
    for pitch in range(num_pitches):
        # Trova dove le note iniziano e finiscono
        active = False
        onset = -1
        
        for frame in range(num_frames):
            if tensor[pitch, frame] >= threshold and not active:
                # Inizio nuova nota
                active = True
                onset = frame
            elif tensor[pitch, frame] < threshold and active:
                # Fine nota
                active = False
                notes.append((pitch, onset, frame))
                onset = -1
        
        # Se una nota è ancora attiva alla fine
        if active:
            notes.append((pitch, onset, num_frames - 1))
    
    return notes

def evaluate_note_prediction(
    yp_gt: torch.Tensor,
    yp_pred: torch.Tensor,
    onset_tol_frames: int = 2,  # tolleranza in frame (non più in secondi)
    duration_tol_ratio: float = 0.2,  # 20% di tolleranza sulla durata
    debug: bool = False,
) -> dict[str, float]:
    """
    Valuta la predizione delle note confrontando con ground truth
    
    Args:
        yp_gt: tensore ground truth (pitch x time)
        yp_pred: tensore predetto (pitch x time)
        threshold: soglia per binarizzazione
        onset_tol_frames: tolleranza onset in frame
        duration_tol_ratio: tolleranza durata (0.2 = 20%)
        debug: se True stampa informazioni di debug
    
    Returns:
        dict con metriche di valutazione
    """
    threshold = s.threshold  # soglia per binarizzazione
    # Assicurati che i tensori abbiano le stesse dimensioni
    assert yp_gt.shape == yp_pred.shape, f"Shape mismatch: GT {yp_gt.shape}, Pred {yp_pred.shape}"
    
    # Binarizza i tensori
    gt_binary = (yp_gt >= threshold).float()
    pred_binary = (yp_pred >= threshold).float()
    
    # Estrai note da GT e predizione
    gt_notes = extract_notes_from_tensor(gt_binary)
    pred_notes = extract_notes_from_tensor(pred_binary)
    
    if debug:
        print(f"Note GT trovate: {len(gt_notes)}")
        print(f"Note Pred trovate: {len(pred_notes)}")
        if gt_notes:
            print(f"Esempio nota GT: pitch={gt_notes[0][0]}, onset={gt_notes[0][1]}, offset={gt_notes[0][2]}")
        if pred_notes:
            print(f"Esempio nota Pred: pitch={pred_notes[0][0]}, onset={pred_notes[0][1]}, offset={pred_notes[0][2]}")
    
    # Inizializza contatori
    true_positives = 0
    matched_gt_indices = set()
    matched_pred_indices = set()
    
    # Per ogni nota predetta, cerca match nel GT
    for pred_idx, (pred_pitch, pred_onset, pred_offset) in enumerate(pred_notes):
        pred_duration = pred_offset - pred_onset
        
        for gt_idx, (gt_pitch, gt_onset, gt_offset) in enumerate(gt_notes):
            if gt_idx in matched_gt_indices:
                continue  # GT già matchato
                
            if pred_pitch != gt_pitch:
                continue  # Pitch diverso
                
            gt_duration = gt_offset - gt_onset
            
            # Check onset match (entro tolleranza)
            onset_diff = abs(pred_onset - gt_onset)
            if onset_diff > onset_tol_frames:
                continue
                
            # Check duration match (entro tolleranza percentuale)
            duration_diff = abs(pred_duration - gt_duration)
            max_duration = max(pred_duration, gt_duration, 1)  # evita divisione per zero
            if duration_diff > duration_tol_ratio * max_duration:
                continue
                
            # Match trovato!
            true_positives += 1
            matched_gt_indices.add(gt_idx)
            matched_pred_indices.add(pred_idx)
            break
    
    # Calcola metriche
    false_positives = len(pred_notes) - len(matched_pred_indices)
    false_negatives = len(gt_notes) - len(matched_gt_indices)
    
    precision = true_positives / (true_positives + false_positives + 1e-9)
    recall = true_positives / (true_positives + false_negatives + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    
    # Accuratezza per nota (percentuale di note corrette)
    note_accuracy = true_positives / len(gt_notes) if gt_notes else 0
    
    if debug:
        print(f"\n=== RISULTATI VALUTAZIONE ===")
        print(f"Note totali GT: {len(gt_notes)}")
        print(f"Note totali Predette: {len(pred_notes)}")
        print(f"True Positives (note corrette): {true_positives}")
        print(f"False Positives (note extra): {false_positives}")
        print(f"False Negatives (note mancanti): {false_negatives}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Note Accuracy: {note_accuracy:.4f}")
        print(f"Note corrette: {true_positives}/{len(gt_notes)}")
    

    return {
        "TP": true_positives,
        "FP": false_positives,
        "FN": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }

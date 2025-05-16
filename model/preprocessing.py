import numpy as np
import librosa
import torch

from settings import Settings


# ---------- CQT ----------
"""
Args:
    wav_file: path to the audio file
    sr: sampling rate
    hop_length: hop length for CQT
    n_bins: number of bins for CQT
Returns:
    cqt_tensor: CQT tensor of shape (n_bins, time_steps)
"""
def constant_q_transform(y: np.ndarray, sr: int, hop_length: int, n_bins: int) -> np.ndarray:
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()

    # Ora y è un array NumPy, quindi possiamo passarlo a librosa
    cqt = librosa.cqt(y, sr=sr, hop_length=hop_length, n_bins=n_bins)
    return np.abs(cqt)

# ---------- Harmonic Stacking ----------
"""
Args:
    cqt_tensor: CQT tensor of shape (n_bins, time_steps)
    shifts: list of shifts to apply
Returns:
    stacked_tensor: stacked tensor of shape (len(shifts), n_bins, time_steps)
"""
def harmonic_stacking(cqt_np: np.ndarray, shifts: list[int]) -> np.ndarray:
    stacked = []
    for shift in shifts:
        shifted = np.roll(cqt_np, shift, axis=0)
        if shift > 0:
            shifted[:shift, :] = 0
        elif shift < 0:
            shifted[shift:, :] = 0
        stacked.append(shifted)
    return np.stack(stacked, axis=0)


# ---------- Preprocessing Function ----------
def preprocess(y: np.ndarray) -> torch.Tensor:
    
    batch = []
    
    for i in range(len(y)):
        
        cqt_tensor = constant_q_transform(y[i], Settings.sample_rate, Settings.hop_length, Settings.n_bins)
        stacked_tensor = harmonic_stacking(cqt_tensor, Settings.harmonic_shifts)
        input_tensor = torch.tensor(stacked_tensor).float().to(Settings.device)   # Add batch dimension
        batch.append(input_tensor)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = torch.stack(batch, dim=0).to(device)  # Stack the batch and move to device
    
    return batch
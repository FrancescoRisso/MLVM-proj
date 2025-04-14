import tensorflow as tf
import numpy as np
import librosa


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
def constant_q_transform(wav_file: str, sr: int, hop_length: int, n_bins: int) -> tf.Tensor:
    y, sr = librosa.load(wav_file, sr=sr)
    cqt = librosa.cqt(y, sr=sr, hop_length=hop_length, n_bins=n_bins)
    return tf.convert_to_tensor(np.abs(cqt), dtype=tf.float32)


# ---------- Harmonic Stacking ----------
"""
Args:
    cqt_tensor: CQT tensor of shape (n_bins, time_steps)
    shifts: list of shifts to apply
Returns:
    stacked_tensor: stacked tensor of shape (len(shifts), n_bins, time_steps)
"""
def harmonic_stacking(cqt_tensor: tf.Tensor, shifts: list[int]) -> tf.Tensor:
    stacked = []
    cqt_np = cqt_tensor.numpy()
    for shift in shifts:
        shifted = np.roll(cqt_np, shift, axis=0)
        if shift > 0:
            shifted[:shift, :] = 0
        elif shift < 0:
            shifted[shift:, :] = 0
        stacked.append(shifted)
    stacked_tensor = tf.convert_to_tensor(np.stack(stacked, axis=0), dtype=tf.float32)
    return stacked_tensor

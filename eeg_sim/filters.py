import numpy as np
from scipy.signal import butter, filtfilt

# --- filter design helpers ---

def butter_bandpass(l_freq: float, h_freq: float, fs: float, order: int = 4):
    nyq = 0.5 * fs
    lo = l_freq / nyq
    hi = h_freq / nyq
    if lo <= 0 or hi >= 1 or lo >= hi:
        raise ValueError('Invalid band edges. Ensure 0 < l_freq < h_freq < fs/2.')
    b, a = butter(order, [lo, hi], btype='bandpass')
    return b, a


def butter_highpass(cutoff: float, fs: float, order: int = 2):
    nyq = 0.5 * fs
    w = cutoff / nyq
    if w <= 0 or w >= 1:
        raise ValueError('Invalid cutoff. Ensure 0 < cutoff < fs/2.')
    b, a = butter(order, w, btype='highpass')
    return b, a


def butter_lowpass(cutoff: float, fs: float, order: int = 4):
    nyq = 0.5 * fs
    w = cutoff / nyq
    if w <= 0 or w >= 1:
        raise ValueError('Invalid cutoff. Ensure 0 < cutoff < fs/2.')
    b, a = butter(order, w, btype='lowpass')
    return b, a


def apply_filter(x: np.ndarray, b: np.ndarray, a: np.ndarray) -> np.ndarray:
    # Zero-phase filtering (no phase distortion) like offline preprocessing
    return filtfilt(b, a, x)


def amplify(x: np.ndarray, gain: float) -> np.ndarray:
    if not np.isfinite(gain):
        raise ValueError('Gain must be a finite value.')
    return gain * x

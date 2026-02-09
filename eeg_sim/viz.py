import matplotlib.pyplot as plt
from scipy.signal import welch, spectrogram
import numpy as np

# --- plotting helpers ---

def plot_time(t, signals, title, ylabel='Amplitude (uV)', xlim=None):
    plt.figure(figsize=(10, 4))
    for name, y in signals.items():
        plt.plot(t, y, label=name, linewidth=1.0)
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel(ylabel)
    if xlim is not None:
        plt.xlim(xlim)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right', fontsize=9)
    plt.tight_layout()


def plot_psd(x, fs, title, nperseg=1024):
    f, pxx = welch(x, fs=fs, nperseg=min(nperseg, len(x)))
    plt.figure(figsize=(10, 4))
    plt.semilogy(f, pxx)
    plt.title(title)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD (uV^2/Hz)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def plot_spectrogram(x, fs, title):
    f, t, Sxx = spectrogram(x, fs=fs, nperseg=256, noverlap=192)
    plt.figure(figsize=(10, 4))
    plt.pcolormesh(t, f, 10*np.log10(Sxx + 1e-20), shading='auto')
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.ylim(0, 40)
    plt.colorbar(label='Power (dB)')
    plt.tight_layout()

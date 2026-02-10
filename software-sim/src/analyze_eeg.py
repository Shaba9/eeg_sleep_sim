
import argparse
import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch
import mne

# -------------------------
# Helpers
# -------------------------

def bandpass_filter(raw, low, high):
    raw.filter(l_freq=low, h_freq=high, picks='eeg', verbose=False)
    return raw

def apply_notch(raw, notch_hz):
    if notch_hz is None:
        return raw
    raw.notch_filter(freqs=[notch_hz], picks='eeg', verbose=False)
    return raw

def resample_if_needed(raw, fs_target):
    if fs_target is None:
        return raw
    raw.resample(fs_target, npad='auto', verbose=False)
    return raw

def detect_eeg_channels(ch_names, include_substrings):
    if include_substrings:
        sel = [i for i, ch in enumerate(ch_names) if any(s.lower() in ch.lower() for s in include_substrings)]
    else:
        scalp_keywords = ["EEG", "Fp", "Fz", "F", "C", "P", "O", "T", "A1", "A2", "M1", "M2"]
        sel = [i for i, ch in enumerate(ch_names) if any(k.lower() in ch.lower() for k in scalp_keywords)]
    return sel if sel else list(range(len(ch_names)))


def compute_welch(x, fs, win_sec=4.0, overlap=0.5):
    nperseg = int(win_sec * fs)
    noverlap = int(nperseg * overlap)
    f, pxx = welch(x, fs=fs, nperseg=nperseg, noverlap=noverlap, detrend='constant')
    return f, pxx


def bandpower_from_psd(f, pxx, band):
    lo, hi = band
    mask = (f >= lo) & (f < hi)
    if not np.any(mask):
        return np.nan
    return np.trapezoid(pxx[mask], f[mask])

# -------------------------
# Main
# -------------------------
parser = argparse.ArgumentParser(description='EEG Analysis Pipeline (EDF or CSV) — Manual Data Version')
parser.add_argument('--input', required=True, help='Path to EDF or CSV')
parser.add_argument('--format', required=True, choices=['edf','csv'], help='Input format')
parser.add_argument('--config', default='config.yaml', help='Config YAML path')
args = parser.parse_args()

cfg = yaml.safe_load(open(args.config))
out_dir = Path(cfg['io']['output_dir'])
out_dir.mkdir(parents=True, exist_ok=True)

if args.format == 'edf':
    raw = mne.io.read_raw_edf(args.input, preload=True, verbose=False)
    ch_names = raw.info['ch_names']
    sel_idx = detect_eeg_channels(ch_names, cfg['io']['channel_include'])
    picks = [ch_names[i] for i in sel_idx]
    raw.pick(picks)
    fs = raw.info['sfreq']

    raw = bandpass_filter(raw, cfg['filter']['bandpass']['low_hz'], cfg['filter']['bandpass']['high_hz'])
    raw = apply_notch(raw, cfg['filter']['notch_hz'])
    raw = resample_if_needed(raw, cfg['filter']['resample_hz'])
    fs = raw.info['sfreq']

    data = raw.get_data()
    ch_names = raw.info['ch_names']

elif args.format == 'csv':
    df = pd.read_csv(args.input)
    df_num = df.select_dtypes(include=['number'])
    data = df_num.to_numpy().T
    ch_names = list(df_num.columns)
    fs = None
    if 'time' in df.columns:
        t = df['time'].to_numpy()
        dt = np.median(np.diff(t))
        if dt > 0:
            fs = 1.0/dt
else:
    raise ValueError('format must be edf or csv')

# ----------- Raw & filtered excerpts (EDF only) -----------
if args.format == 'edf':
    seconds = 60
    n_samples = int(seconds * fs)
    if data.shape[1] < n_samples:
        n_samples = data.shape[1]
    t = np.arange(n_samples) / fs
    ch0 = data[0, :n_samples]

    plt.figure(figsize=(12, 4))
    plt.plot(t, ch0, lw=0.7)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (uV)')
    plt.title(f'Raw excerpt (first channel: {ch_names[0]})')
    plt.tight_layout()
    plt.savefig(out_dir / 'raw_excerpt.png', dpi=150)
    plt.close()

    plt.figure(figsize=(12, 4))
    plt.plot(t, ch0, lw=0.7, color='tab:green')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (uV)')
    plt.title(f'Filtered excerpt (first channel: {ch_names[0]})')
    plt.tight_layout()
    plt.savefig(out_dir / 'filtered_excerpt.png', dpi=150)
    plt.close()

# ----------- PSD + Bandpowers -----------
psd_fig = out_dir / 'psd.png'
features_csv = out_dir / 'features_bandpower.csv'
summary_txt = out_dir / 'summary_report.txt'

features_rows = []

if fs is not None and fs > 0:
    plt.figure(figsize=(10, 6))
    for i, ch in enumerate(ch_names):
        f, pxx = compute_welch(data[i], fs, win_sec=cfg['analysis']['window_sec'], overlap=cfg['analysis']['overlap'])
        plt.semilogy(f, pxx, alpha=0.6, label=ch)
        for band_name, band_range in cfg['analysis']['bands'].items():
            bp = bandpower_from_psd(f, pxx, band_range)
            features_rows.append({'channel': ch, 'band': band_name, 'low_hz': band_range[0], 'high_hz': band_range[1], 'power': bp})
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD (V^2/Hz)')
    plt.title('Power Spectral Density (Welch)')
    plt.legend(loc='upper right', ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(psd_fig, dpi=150)
    plt.close()

    if features_rows:
        pd.DataFrame(features_rows).to_csv(features_csv, index=False)

    # Spectrogram for one channel
    ch_idx = int(cfg['analysis']['spectrogram_channel'])
    ch_idx = max(0, min(ch_idx, len(ch_names)-1))
    x = data[ch_idx]
    plt.figure(figsize=(11, 4))
    NFFT = int(cfg['analysis']['window_sec'] * fs)
    noverlap = int(cfg['analysis']['overlap'] * NFFT)
    Pxx, freqs, bins, im = plt.specgram(x, NFFT=NFFT, Fs=fs, noverlap=noverlap, cmap='viridis')
    plt.colorbar(label='Power (dB)')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title(f'Spectrogram (channel: {ch_names[ch_idx]})')
    plt.ylim(0, max(45, min(60, fs/2)))
    plt.tight_layout()
    plt.savefig(out_dir / 'spectrogram.png', dpi=150)
    plt.close()

# ----------- Summary -----------
with open(summary_txt, 'w') as f:
    f.write('EEG Analysis Summary')
    f.write('====================')
    f.write(f"Input: {args.input}")
    f.write(f"Format: {args.format}")
    if fs is not None:
        f.write(f"Sampling rate (Hz): {fs}")
    f.write(f"Channels used: {', '.join(ch_names)}")
    f.write("Filters:")
    f.write(f"  Bandpass: {cfg['filter']['bandpass']['low_hz']}–{cfg['filter']['bandpass']['high_hz']} Hz")
    f.write(f"  Notch: {cfg['filter']['notch_hz']} Hz")
    f.write(f"  Resample: {cfg['filter']['resample_hz']}")
    f.write("Outputs written to: {}".format(out_dir.resolve()))

print("Done. Check the 'outputs' folder for plots and features.")


import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from scipy.signal import welch, spectrogram
import mne


# -------------------------
# Helpers
# -------------------------

def bandpass_filter(raw: mne.io.BaseRaw, low: float, high: float) -> mne.io.BaseRaw:
    """In-place band-pass on EEG channels."""
    raw.filter(l_freq=low, h_freq=high, picks='eeg', verbose=False)
    return raw


def apply_notch(raw: mne.io.BaseRaw, notch_hz) -> mne.io.BaseRaw:
    """In-place notch filter; accepts None, a number, or a list/tuple of numbers."""
    if notch_hz in (None, 'None', 'null'):
        return raw
    if isinstance(notch_hz, (list, tuple)):
        freqs = list(notch_hz)
    else:
        freqs = [float(notch_hz)]
    if len(freqs) > 0:
        raw.notch_filter(freqs=freqs, picks='eeg', verbose=False)
    return raw


def resample_if_needed(raw: mne.io.BaseRaw, fs_target) -> mne.io.BaseRaw:
    """In-place resample if fs_target provided."""
    if fs_target in (None, 'None', 'null'):
        return raw
    raw.resample(float(fs_target), npad='auto', verbose=False)
    return raw


def detect_eeg_channels(ch_names, include_substrings):
    """
    Detect EEG channels.
    - If include_substrings provided, use them (case-insensitive substring match).
    - Else, use a regex that matches common EEG channel patterns.
      Falls back to all channels if nothing matches.
    """
    if include_substrings:
        sel = [i for i, ch in enumerate(ch_names)
               if any(s.lower() in ch.lower() for s in include_substrings)]
        return sel if sel else list(range(len(ch_names)))

    # Regex pattern includes common EEG labels and variants
    # Examples: Fp1, F3, C3, Pz, Oz, T7, A1, M2, "EEG F3-Ref"
    pat = re.compile(r'\b(?:EEG|Fpz?|Fz?|Cz?|Pz?|Oz?|T[3-8]|A[12]|M[12]|[FOCTP]\d+)\b', re.I)
    sel = [i for i, ch in enumerate(ch_names) if pat.search(ch)]
    return sel if sel else list(range(len(ch_names)))


def compute_welch(x, fs, win_sec=4.0, overlap=0.5):
    nperseg = int(win_sec * fs)
    noverlap = int(nperseg * overlap)
    f, pxx = welch(x, fs=fs, nperseg=nperseg, noverlap=noverlap, detrend='constant')
    return f, pxx


# Provide compatibility with older NumPy versions
try:
    _trapz_func = np.trapezoid  # NumPy >= 1.20
except AttributeError:
    _trapz_func = np.trapz


def bandpower_from_psd(f, pxx, band):
    lo, hi = band
    mask = (f >= lo) & (f < hi)
    if not np.any(mask):
        return np.nan
    return _trapz_func(pxx[mask], f[mask])


def load_config(path: Path) -> dict:
    with open(path, 'r') as fh:
        return yaml.safe_load(fh)


def ensure_outdir(cfg: dict) -> Path:
    out_dir = Path(cfg['io']['output_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def read_edf(input_path: Path, cfg: dict):
    """Load EDF, pick EEG channels, capture pre/post filter data, return arrays and metadata."""
    raw = mne.io.read_raw_edf(str(input_path), preload=True, verbose=False)
    ch_names_all = raw.info['ch_names']

    # Channel selection
    sel_idx = detect_eeg_channels(ch_names_all, cfg['io'].get('channel_include', []))
    picks = [ch_names_all[i] for i in sel_idx]
    raw.pick(picks)

    # Snapshot BEFORE filtering for raw excerpt
    data_before = raw.copy().get_data()
    fs_before = float(raw.info['sfreq'])
    ch_names = raw.info['ch_names']

    # Filtering chain (apply notch, then bandpass, then optional resample)
    notch = cfg['filter'].get('notch_hz', None)
    raw = apply_notch(raw, notch)
    bp = cfg['filter']['bandpass']
    raw = bandpass_filter(raw, bp['low_hz'], bp['high_hz'])
    raw = resample_if_needed(raw, cfg['filter'].get('resample_hz', None))

    # AFTER filtering
    data_after = raw.get_data()
    fs = float(raw.info['sfreq'])
    ch_names_after = raw.info['ch_names']

    return {
        'data_before': data_before,
        'data': data_after,
        'fs_before': fs_before,
        'fs': fs,
        'ch_names': ch_names_after
    }


def read_csv(input_path: Path):
    """
    Load CSV, use numeric columns as channels (rows = samples), transpose to (n_channels, n_samples).
    Attempt to infer fs if 'time' column exists (seconds).
    """
    df = pd.read_csv(input_path)
    df_num = df.select_dtypes(include=['number'])
    if df_num.empty:
        raise ValueError("No numeric columns found in CSV.")

    data = df_num.to_numpy().T
    ch_names = list(df_num.columns)
    fs = None

    # Try to infer sampling rate if 'time' column present (case-insensitive)
    time_col = None
    for c in df.columns:
        if c.lower() == 'time':
            time_col = c
            break
    if time_col is not None:
        t = df[time_col].to_numpy()
        if len(t) > 1:
            dt = np.median(np.diff(t))
            if dt > 0:
                fs = 1.0 / dt

    return {
        'data': data,
        'fs': fs,
        'ch_names': ch_names
    }


def plot_time_excerpt(signal, fs, ch_name, out_path: Path, title: str, seconds: int = 60, color=None):
    n_samples = int(seconds * fs)
    n_samples = min(n_samples, signal.shape[-1])
    t = np.arange(n_samples) / fs
    plt.figure(figsize=(12, 4))
    plt.plot(t, signal[:n_samples], lw=0.7, color=color)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (signal units)')
    plt.title(f'{title} (first channel: {ch_name})')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_psd_and_bandpowers(data, fs, ch_names, cfg, out_dir: Path):
    psd_fig = out_dir / 'psd.png'
    features_csv = out_dir / 'features_bandpower.csv'
    features_rows = []

    plt.figure(figsize=(10, 6))
    for i, ch in enumerate(ch_names):
        f, pxx = compute_welch(
            data[i],
            fs,
            win_sec=cfg['analysis']['window_sec'],
            overlap=cfg['analysis']['overlap']
        )
        plt.semilogy(f, pxx, alpha=0.6, label=ch)

        for band_name, band_range in cfg['analysis']['bands'].items():
            bp = bandpower_from_psd(f, pxx, band_range)
            features_rows.append({
                'channel': ch,
                'band': band_name,
                'low_hz': band_range[0],
                'high_hz': band_range[1],
                'power': bp
            })

    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD (signal units²/Hz)')
    plt.title('Power Spectral Density (Welch)')
    plt.legend(loc='upper right', ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(psd_fig, dpi=150)
    plt.close()

    if features_rows:
        pd.DataFrame(features_rows).to_csv(features_csv, index=False)

    return psd_fig, features_csv


def plot_spectrogram(x, fs, ch_name, cfg, out_dir: Path):
    # Parameters consistent with Welch windows
    nperseg = int(cfg['analysis']['window_sec'] * fs)
    noverlap = int(cfg['analysis']['overlap'] * nperseg)
    nperseg = max(16, nperseg)  # guard small windows
    noverlap = min(noverlap, nperseg - 1) if nperseg > 1 else 0

    f, t, Sxx = spectrogram(x, fs=fs, nperseg=nperseg, noverlap=noverlap, detrend='constant', scaling='density', mode='psd')
    Sxx_db = 10 * np.log10(np.maximum(Sxx, 1e-20))

    plt.figure(figsize=(11, 4))
    mesh = plt.pcolormesh(t, f, Sxx_db, shading='gouraud', cmap='viridis')
    cbar = plt.colorbar(mesh)
    cbar.set_label('Power (dB)')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title(f'Spectrogram (channel: {ch_name})')

    # Limit to a practical EEG range (up to 60 Hz and below Nyquist)
    plt.ylim(0, max(45, min(60, fs / 2)))
    plt.tight_layout()
    out_path = out_dir / 'spectrogram.png'
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def write_summary(out_dir: Path, args, fs, ch_names, cfg):
    summary_txt = out_dir / 'summary_report.txt'
    lines = []
    lines.append("EEG Analysis Summary")
    lines.append("====================")
    lines.append(f"Input: {args.input}")
    lines.append(f"Format: {args.format}")
    if fs is not None:
        lines.append(f"Sampling rate (Hz): {fs:.4f}")
    lines.append(f"Channels used: {', '.join(ch_names)}")
    lines.append("Filters:")
    lines.append(f"  Bandpass: {cfg['filter']['bandpass']['low_hz']}–{cfg['filter']['bandpass']['high_hz']} Hz")
    lines.append(f"  Notch: {cfg['filter'].get('notch_hz', None)} Hz")
    lines.append(f"  Resample: {cfg['filter'].get('resample_hz', None)}")
    lines.append(f"Outputs written to: {out_dir.resolve()}")

    with open(summary_txt, 'w') as f:
        f.write("\n".join(lines) + "\n")

    return summary_txt


# -------------------------
# Main
# -------------------------

def main():
    parser = argparse.ArgumentParser(description='EEG Analysis Pipeline (EDF or CSV) — Manual Data Version (Refactored)')
    parser.add_argument('--input', required=True, help='Path to EDF or CSV')
    parser.add_argument('--format', required=True, choices=['edf', 'csv'], help='Input format')
    parser.add_argument('--config', default='config.yaml', help='Config YAML path')
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    out_dir = ensure_outdir(cfg)

    if args.format == 'edf':
        payload = read_edf(Path(args.input), cfg)
        data_before = payload['data_before']
        data = payload['data']
        fs_before = payload['fs_before']
        fs = payload['fs']
        ch_names = payload['ch_names']

        # --- Raw/Filtered excerpts (first channel) ---
        if data.shape[0] > 0:
            ch0_name = ch_names[0]
            plot_time_excerpt(
                signal=data_before[0], fs=fs_before, ch_name=ch0_name,
                out_path=out_dir / 'raw_excerpt.png', title='Raw excerpt', seconds=60, color=None
            )
            plot_time_excerpt(
                signal=data[0], fs=fs, ch_name=ch0_name,
                out_path=out_dir / 'filtered_excerpt.png', title='Filtered excerpt', seconds=60, color='tab:green'
            )

    elif args.format == 'csv':
        payload = read_csv(Path(args.input))
        data = payload['data']
        ch_names = payload['ch_names']
        fs = payload['fs']

    else:
        raise ValueError("format must be 'edf' or 'csv'")

    # --- PSD + Bandpowers + Spectrogram (only if fs known) ---
    if fs is not None and fs > 0:
        psd_fig, features_csv = plot_psd_and_bandpowers(data, fs, ch_names, cfg, out_dir)

        # Spectrogram for one channel
        ch_idx = int(cfg['analysis']['spectrogram_channel'])
        ch_idx = max(0, min(ch_idx, len(ch_names) - 1))
        spec_path = plot_spectrogram(data[ch_idx], fs, ch_names[ch_idx], cfg, out_dir)
    else:
        print("Warning: Sampling rate (fs) could not be determined. Skipping PSD, bandpowers, and spectrogram.")
        psd_fig = None
        features_csv = None

    # --- Summary ---
    summary_txt = write_summary(out_dir, args, fs, ch_names, cfg)

    print(f"Done. Check '{out_dir.resolve()}' for outputs.")
    if psd_fig:
        print(f"- PSD figure:          {psd_fig}")
    if args.format == 'edf':
        print(f"- Raw excerpt:         {out_dir / 'raw_excerpt.png'}")
        print(f"- Filtered excerpt:    {out_dir / 'filtered_excerpt.png'}")
        print(f"- Spectrogram:         {out_dir / 'spectrogram.png'}")
    if features_csv:
        print(f"- Bandpower features:  {features_csv}")
    print(f"- Summary report:      {summary_txt}")


if __name__ == '__main__':
    main()
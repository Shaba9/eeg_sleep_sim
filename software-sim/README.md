
# EEG Insomnia Lab — Manual Data Version (Sleep‑EDF)

This package provides a ready‑to‑run EEG analysis pipeline for **Sleep‑EDF** recordings with **no automatic downloads**. You will manually place EDF or CSV files in `./data/`, then run preprocessing, PSD, band‑powers, and spectrogram analysis to satisfy lab requirements.

## 📦 Contents
```
./
  README.md               ← this guide
  requirements.txt        ← Python dependencies
  config.yaml             ← Tunable parameters (filters, bands, channels)
  src/
    convert_edf_to_csv.py ← Convert EDF to CSV using MNE (optional)
    analyze_eeg.py        ← Preprocess + PSD + band powers + spectrogram + plots
  data/                   ← Put your EDF/CSV here (you provide)
  outputs/                ← Figures and feature tables land here
```

## 🚀 Quick Start

### 1) Create & activate a virtual environment (recommended)
```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Place your data
- Download an EDF file (e.g., `SC4001E0-PSG.edf`) from your chosen source.
- Copy it into the `data/` folder.

> If your lab requires CSV, convert EDF → CSV:
```bash
python src/convert_edf_to_csv.py --edf data/SC4001E0-PSG.edf --out data/SC4001E0-PSG.csv
```

### 4) Run the analysis
**Option A – EDF input**
```bash
python src/analyze_eeg.py --input data/SC4001E0-PSG.edf --format edf --config config.yaml
```
**Option B – CSV input**
```bash
python src/analyze_eeg.py --input data/SC4001E0-PSG.csv --format csv --config config.yaml
```

Artifacts will be saved to `./outputs`:
- `raw_excerpt.png` — 60‑s snippet of raw signal
- `filtered_excerpt.png` — same snippet after filters
- `psd.png` — Power Spectral Density per channel (Welch)
- `spectrogram.png` — Time–frequency view (STFT) for one channel
- `features_bandpower.csv` — Band powers per channel (Δ, θ, α, β, γ)
- `summary_report.txt` — Sampling rate, filters, notes

## 🧪 Lab Discussion
- **Time domain**: Compare `raw_excerpt.png` vs `filtered_excerpt.png` to discuss amplitude variations and artifact suppression.
- **Frequency content**: Use `psd.png` to identify dominant bands; cite `features_bandpower.csv` quantitatively.
- **Temporal patterns**: Inspect `spectrogram.png` to comment on changes over time (e.g., sleep stage dynamics if you also have hypnograms).
- **Interpretation**: Relate alpha/beta/delta patterns to sleep physiology and discuss limitations (single subject, clinical vs. lab context).

## ⚙️ Config (`config.yaml`)
- `filter.bandpass`: default 0.5–40 Hz
- `filter.notch_hz`: 60 Hz (set 50 Hz where applicable)
- `filter.resample_hz`: optional downsample for speed
- `analysis.window_sec`, `analysis.overlap`: Welch/STFT params
- `analysis.bands`: editable band edges
- `io.channel_include`: restrict channels by name substring

## 🧰 Tips
- If sampling rate is very high, set `filter.resample_hz` (e.g., 100) to speed up plots.
- For CSV inputs without a known sampling rate, PSD is skipped unless a time column is present to infer `fs`.

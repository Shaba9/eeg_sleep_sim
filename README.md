# EEG Sleep-Band Simulation (Python)

This mini-project simulates an EEG-like signal relevant to sleep staging and sleep disorders, then demonstrates amplification and filtering to isolate frequency bands (delta/theta/sigma). It produces time-domain plots, PSD plots, and an optional spectrogram.

## Files
- `main.py` - runs the simulation, filtering, and plotting
- `eeg_sim/` - helper package
- `requirements.txt` - dependencies
- `LAB_REPORT_NOTES.txt` - quick mapping to a lab report narrative

## Quick start
```bash
pip install -r requirements.txt
python main.py
```

## What the script does
- Generates a synthetic EEG mixture with sleep-relevant components:
  - Delta: 0.5-4 Hz (deep sleep / N3)
  - Theta: 4-7 Hz (drowsiness / N1)
  - Sigma (spindle band): 11-16 Hz (N2 spindles)
- Adds baseline noise and optional artifacts (eye blink, 60 Hz line noise)
- Applies an amplifier gain (e.g., 1000x)
- Filters into bands using Butterworth band-pass filters
- Plots raw vs amplified vs filtered signals and their spectra

## Customize
Edit `main.py` to change sampling rate (FS), duration, gain, filter bands, and artifact toggles.

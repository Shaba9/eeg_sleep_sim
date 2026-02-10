
import argparse
import pandas as pd
import mne

parser = argparse.ArgumentParser(description="Convert EDF to CSV using MNE")
parser.add_argument('--edf', required=True, help='Path to EDF file')
parser.add_argument('--out', required=True, help='Output CSV path')
args = parser.parse_args()

raw = mne.io.read_raw_edf(args.edf, preload=True, verbose=False)
df = raw.to_data_frame()
df.to_csv(args.out, index=False)
print(f"Wrote CSV: {args.out} with shape {df.shape}")

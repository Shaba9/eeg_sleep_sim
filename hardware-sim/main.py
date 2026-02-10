"""EEG Sleep-Band Simulation (Python)

Generates a synthetic EEG-like signal (microvolt scale), applies an amplifier gain,
filters into sleep-relevant bands, and plots results.

Bands:
- Delta: 0.5-4 Hz (deep sleep / N3)
- Theta: 4-7 Hz (drowsiness / N1)
- Sigma: 11-16 Hz (sleep spindles / N2)

Sleep disorder connections:
- Sleep apnea: fragmented sleep reduces consolidated delta; arousals increase mixed-frequency activity.
- Insomnia: sigma/spindle band metrics can quantify sleep stability.
- Narcolepsy/REM dysregulation: time-varying spectral patterns; spectrograms help visualize transitions.
"""

from eeg_sim.signals import make_synthetic_sleep_eeg
from eeg_sim.filters import butter_highpass, butter_lowpass, butter_bandpass, apply_filter, amplify
from eeg_sim.viz import plot_time, plot_psd, plot_spectrogram
import matplotlib.pyplot as plt

# -------------------- User settings --------------------
FS = 250.0          # sampling frequency (Hz)
DURATION = 10.0     # seconds
GAIN = 1000.0       # amplifier gain
INCLUDE_ARTIFACTS = True

# Front-end style filtering (common in EEG hardware)
HP_CUTOFF = 0.5     # Hz
LP_CUTOFF = 40.0    # Hz

# Sleep band definitions
BANDS = {
    'Delta (0.5-4 Hz)': (0.5, 4.0),
    'Theta (4-7 Hz)': (4.0, 7.0),
    'Sigma/Spindle (11-16 Hz)': (11.0, 16.0),
}
# -------------------------------------------------------


def main():
    # 1) Generate synthetic EEG in microvolts
    t, eeg_uV, _ = make_synthetic_sleep_eeg(
        fs=FS, duration_s=DURATION, seed=7, include_artifacts=INCLUDE_ARTIFACTS
    )

    # 2) Amplify (hardware preamp equivalent)
    eeg_amp_uV = amplify(eeg_uV, GAIN)

    # 3) Apply broad front-end filtering (HP + LP)
    b_hp, a_hp = butter_highpass(HP_CUTOFF, FS, order=2)
    b_lp, a_lp = butter_lowpass(LP_CUTOFF, FS, order=4)
    eeg_front_uV = apply_filter(apply_filter(eeg_amp_uV, b_hp, a_hp), b_lp, a_lp)

    # 4) Band-pass filters for sleep-relevant bands
    band_outputs = {}
    for name, (lo, hi) in BANDS.items():
        b, a = butter_bandpass(lo, hi, FS, order=4)
        band_outputs[name] = apply_filter(eeg_front_uV, b, a)

    # 5) Time-domain plots
    plot_time(t, {'Raw EEG (uV)': eeg_uV}, 'Raw Composite EEG (synthetic)', ylabel='Amplitude (uV)', xlim=(0, 5))
    plot_time(t, {'Amplified + front-end filtered (uV)': eeg_front_uV},
              f'After Gain ({GAIN:.0f}x) + HP({HP_CUTOFF} Hz) + LP({LP_CUTOFF} Hz)',
              ylabel='Amplitude (uV)', xlim=(0, 5))
    plot_time(t, band_outputs, 'Band-Isolated Signals (Sleep Bands)', ylabel='Amplitude (uV)', xlim=(0, 5))

    # 6) PSD plots
    plot_psd(eeg_uV, FS, 'PSD of Raw EEG (synthetic)')
    plot_psd(eeg_front_uV, FS, 'PSD After Amplification + Front-end Filtering')

    # 7) Optional spectrogram
    plot_spectrogram(eeg_front_uV, FS, 'Spectrogram (0-40 Hz)')

    plt.show()


if __name__ == '__main__':
    main()

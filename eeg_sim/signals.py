import numpy as np

# --- signal helpers ---

def make_time_vector(duration_s: float, fs: float) -> np.ndarray:
    n = int(np.floor(duration_s * fs))
    return np.arange(n) / fs


def sine_component(t: np.ndarray, freq_hz: float, amp_v: float, phase_rad: float = 0.0) -> np.ndarray:
    return amp_v * np.sin(2 * np.pi * freq_hz * t + phase_rad)


def gaussian_noise(n: int, std_v: float, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(loc=0.0, scale=std_v, size=n)


def make_synthetic_sleep_eeg(
    fs: float = 250.0,
    duration_s: float = 10.0,
    seed: int = 7,
    include_artifacts: bool = True,
):
    """Return (t, eeg_uV, components_uV).

    EEG is synthesized as a mixture of:
    - Delta (2 Hz), Theta (6 Hz), Sigma carrier (13 Hz)
    - Baseline noise
    - Optional eye blink pulses + 60 Hz interference

    Output is in microvolts for convenience.
    """
    rng = np.random.default_rng(seed)
    t = make_time_vector(duration_s, fs)
    n = len(t)

    # Sleep-related components (microvolt scale before conversion)
    delta = sine_component(t, 2.0, 30e-6)   # deep sleep (N3)
    theta = sine_component(t, 6.0, 15e-6)   # drowsy/light sleep (N1)
    sigma = sine_component(t, 13.0, 20e-6)  # spindle band carrier (N2)

    # Baseline noise (instrument + biological background)
    noise = gaussian_noise(n, std_v=5e-6, rng=rng)

    # Optional artifacts
    eye_blink = np.zeros(n)
    line60 = np.zeros(n)
    if include_artifacts:
        # Eye blink: half-sine pulses (large low-frequency transient)
        blink_amp = 120e-6
        blink_times = [2.0, 6.5]
        blink_dur = 0.35
        for bt in blink_times:
            i0 = int(bt * fs)
            i1 = min(n, i0 + int(blink_dur * fs))
            L = i1 - i0
            if L > 2:
                pulse = np.sin(np.linspace(0, np.pi, L))
                eye_blink[i0:i1] += blink_amp * pulse

        # Powerline interference: 60 Hz
        line60 = sine_component(t, 60.0, 8e-6)

    eeg = delta + theta + sigma + noise + eye_blink + line60

    # Convert to microvolts
    to_uV = 1e6
    components_uV = {
        'delta_uV': delta * to_uV,
        'theta_uV': theta * to_uV,
        'sigma_uV': sigma * to_uV,
        'noise_uV': noise * to_uV,
        'eye_blink_uV': eye_blink * to_uV,
        'line60_uV': line60 * to_uV,
    }

    return t, eeg * to_uV, components_uV

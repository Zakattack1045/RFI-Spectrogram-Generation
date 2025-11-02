import numpy as np

def generate_cw_tone(n_samples: int, fs: float, freq_offset: float, phase: float = 0.0) -> np.ndarray:
    """
    Continuous Wave (CW) tone jammer.

    Parameters
    ----------
    n_samples : int
        Number of samples
    fs : float
        Sample rate (Hz)
    freq_offset : float
        Frequency offset from baseband (Hz)
    phase : float
        Initial phase (radians)

    Returns
    -------
    np.ndarray
        Complex jammer signal
    """
    # Use float64 for time array to avoid precision loss with large n_samples
    t = np.arange(n_samples, dtype=np.float64) / fs
    return np.exp(1j * (2 * np.pi * freq_offset * t + phase)).astype(np.complex64)

def generate_pulsed_jammer(
    n_samples: int,
    fs: float,
    freq_offset: float,
    pulse_width: float,
    pulse_period: float,
    phase: float = 0.0
) -> np.ndarray:
    """
    Pulsed jammer - periodic on/off keyed interference.

    Parameters
    ----------
    pulse_width : float
        Pulse duration (seconds)
    pulse_period : float
        Time between pulse starts (seconds)
    """
    # Generate carrier
    carrier = generate_cw_tone(n_samples, fs, freq_offset, phase)

    # Generate pulse train - use float64 for precision
    t = np.arange(n_samples, dtype=np.float64) / fs
    pulse_train = ((t % pulse_period) < pulse_width).astype(np.float32)

    return carrier * pulse_train

# Sawtooth chirip jammer with period T
# y = a * exp (2*j*pi * (fc * t +/- 1/(2T) * BW *  t^2) + phi)
def generate_chirp_jammer(
    n_samples: int,
    fs: float,
    chirp_bw: float,
    chirp_duration: float,
    center_freq: float = 0.0,
    amplitude: float = 1.0
) -> np.ndarray:
    """
    Linear chirp (frequency sweep) jammer.

    Parameters
    ----------
    chirp_bw : float
        Chirp bandwidth (Hz) - sweeps from -bw/2 to +bw/2
    chirp_duration : float
        Duration of one chirp (seconds), then repeats
    center_freq : float
        Center frequency offset (Hz)
    """
    # Use float64 for time array to avoid precision loss with large n_samples
    t = np.arange(n_samples, dtype=np.float64) / fs
    T = chirp_duration / 2
    chirp_rate = chirp_bw / chirp_duration  # Hz/s
    f_start = center_freq - chirp_bw / 2

    # Modulo time within chirp period
    t_mod = t % chirp_duration

    # Instantaneous frequency: f(t) = f_start + chirp_rate * t_mod
    # Phase: integral of 2*pi*f(t) dt = 2*pi*(f_start*t + 0.5*chirp_rate*t^2)
    # But we need to handle phase continuity at boundaries

    # Number of complete chirps before each sample
    n_chirps = np.floor(t / chirp_duration).astype(np.int32)

    # Phase within current chirp
    phase_in_chirp = 2 * np.pi * (f_start * t_mod + 0.5 * chirp_rate * t_mod**2)

    # Add phase offset for each complete chirp to maintain continuity
    # Phase at end of one chirp period
    phase_per_chirp = 2 * np.pi * (f_start * chirp_duration + 0.5 * chirp_rate * chirp_duration**2)
    phase = phase_in_chirp + n_chirps * phase_per_chirp
    # No need for phase modulo - exp() handles any phase value

    return np.exp(1j * phase).astype(np.complex64)

def generate_barrage_jammer(n_samples: int, fs: float, seed: int = None) -> np.ndarray:
    """
    Barrage jammer - wideband Gaussian noise across entire spectrum.
    
    Parameters
    ----------
    seed : int, optional
        Random seed for reproducibility
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Complex Gaussian noise (standard normal for both I and Q)
    I = np.random.randn(n_samples).astype(np.float32)
    Q = np.random.randn(n_samples).astype(np.float32)
    
    # Normalize to unit power
    noise = (I + 1j * Q) / np.sqrt(2.0)
    
    return noise.astype(np.complex64)

def generate_multitone_jammer(
    n_samples: int,
    fs: float,
    frequencies: list,
    phases: list = None
) -> np.ndarray:
    """
    Multi-tone jammer - sum of multiple CW tones.
    
    Parameters
    ----------
    frequencies : list of float
        List of frequency offsets (Hz)
    phases : list of float, optional
        Initial phases for each tone (radians)
    """
    if phases is None:
        phases = [0.0] * len(frequencies)
    
    signal = np.zeros(n_samples, dtype=np.complex64)
    for freq, phase in zip(frequencies, phases):
        signal += generate_cw_tone(n_samples, fs, freq, phase)
    
    # Normalize by number of tones to maintain power
    return signal / np.sqrt(len(frequencies))


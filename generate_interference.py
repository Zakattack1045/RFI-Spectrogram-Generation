#!/usr/bin/env python3
"""
Inject various interference signals into I/Q binary files.
Supports pulsed, chirp, CW, barrage, and other jammer types.

Examples
--------
# Add pulsed jammer at +100 kHz, 10 dB stronger than signal
python inject_interference.py \
  --input clean.dat --output jammed.dat \
  --fs 4e6 --dtype int16 \
  --jammer pulse --freq-offset 100e3 \
  --jnr 10 --pulse-width 50e-6 --pulse-period 200e-6

# Add chirp jammer sweeping ±500 kHz
python inject_interference.py \
  --input clean.dat --output jammed.dat \
  --fs 4e6 --dtype int16 \
  --jammer chirp --chirp-bw 1e6 --chirp-duration 1e-3 --jnr 15

# Add barrage noise jammer
python inject_interference.py \
  --input clean.dat --output jammed.dat \
  --fs 4e6 --dtype int16 \
  --jammer barrage --jnr 5
"""

import argparse
import numpy as np
from pathlib import Path
from typing import Tuple


# ---------------------------
# I/O utilities
# ---------------------------

DTYPE_MAP = {
    "int16": np.int16,
    "float32": np.float32,
    "int8": np.int8,
}

def read_complex_chunk(
    mm: np.memmap,
    start_complex: int,
    end_complex: int,
    iq_order: str = "IQ",
    dtype: str = "int16",
    scale_to_float: bool = True,
) -> np.ndarray:
    """
    Read a chunk of interleaved I/Q and return complex64 array.
    - iq_order: 'IQ' (I then Q) or 'QI' (Q then I)
    - For int types, scales to [-1, 1) if scale_to_float is True.
    """
    start_real = 2 * start_complex
    end_real = end_complex * 2
    if end_real > mm.size:
        end_real = mm.size
    raw = mm[start_real:end_real]
    if raw.size < 2:
        return np.array([], dtype=np.complex64)

    # reshape into pairs
    pairs = raw.reshape(-1, 2)

    if iq_order.upper() == "IQ":
        I = pairs[:, 0].astype(np.float32)
        Q = pairs[:, 1].astype(np.float32)
    elif iq_order.upper() == "QI":
        Q = pairs[:, 0].astype(np.float32)
        I = pairs[:, 1].astype(np.float32)
    else:
        raise ValueError("iq_order must be 'IQ' or 'QI'.")

    if scale_to_float and dtype in ("int16", "int8"):
        if dtype == "int16":
            I /= 32768.0
            Q /= 32768.0
        elif dtype == "int8":
            I /= 128.0
            Q /= 128.0

    return (I + 1j * Q).astype(np.complex64)

def save_iq_file(path: str, signal: np.ndarray, dtype: np.dtype):
    """Save complex signal back to I/Q binary file."""
    # Scale back to integer range if needed
    I = signal.real
    Q = signal.imag
    
    if dtype == np.int16:
        I = np.clip(I * 32768.0, -32768, 32767)
        Q = np.clip(Q * 32768.0, -32768, 32767)
        I = I.astype(np.int16)
        Q = Q.astype(np.int16)
    elif dtype == np.int8:
        I = np.clip(I * 128.0, -128, 127)
        Q = np.clip(Q * 128.0, -128, 127)
        I = I.astype(np.int8)
        Q = I.astype(np.int8)
    elif dtype == np.float32:
        I = I.astype(np.float32)
        Q = Q.astype(np.float32)
    
    # Interleave I and Q
    interleaved = np.empty(len(I) * 2, dtype=dtype)
    interleaved[0::2] = I
    interleaved[1::2] = Q
    
    interleaved.tofile(path)


# ---------------------------
# Jammer signal generators
# ---------------------------

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
    t = np.arange(n_samples, dtype=np.float32) / fs
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
    
    # Generate pulse train
    t = np.arange(n_samples, dtype=np.float32) / fs
    pulse_train = ((t % pulse_period) < pulse_width).astype(np.float32)
    
    return carrier * pulse_train

def generate_chirp_jammer(
    n_samples: int,
    fs: float,
    chirp_bw: float,
    chirp_duration: float,
    center_freq: float = 0.0
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
    t = np.arange(n_samples, dtype=np.float32) / fs
    t_mod = t % chirp_duration  # Repeating chirp
    
    # Linear chirp: instantaneous frequency = f0 + k*t
    # where k = chirp_rate = bandwidth / duration
    k = chirp_bw / chirp_duration
    f0 = center_freq - chirp_bw / 2
    
    # Instantaneous phase: integral of 2*pi*f(t)
    phase = 2 * np.pi * (f0 * t_mod + 0.5 * k * t_mod**2)
    
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

def generate_swept_tone_jammer(
    n_samples: int,
    fs: float,
    sweep_range: Tuple[float, float],
    sweep_rate: float
) -> np.ndarray:
    """
    Swept tone jammer - slowly sweeping single tone.
    
    Parameters
    ----------
    sweep_range : tuple of (f_min, f_max)
        Frequency range to sweep (Hz)
    sweep_rate : float
        Sweep rate (Hz/second)
    """
    t = np.arange(n_samples, dtype=np.float32) / fs
    f_min, f_max = sweep_range
    
    # Sawtooth sweep that resets
    sweep_period = (f_max - f_min) / sweep_rate
    t_mod = t % sweep_period
    freq = f_min + sweep_rate * t_mod
    
    # Integrate frequency to get phase
    phase = 2 * np.pi * (f_min * t + 0.5 * sweep_rate * t_mod**2)
    
    return np.exp(1j * phase).astype(np.complex64)


# ---------------------------
# Power scaling
# ---------------------------

def compute_signal_power(signal: np.ndarray) -> float:
    """Compute average power of complex signal."""
    return np.mean(np.abs(signal)**2)

def scale_to_jnr(jammer: np.ndarray, signal: np.ndarray, jnr_db: float) -> np.ndarray:
    """
    Scale jammer to achieve specified Jammer-to-Noise Ratio (JNR).
    
    Parameters
    ----------
    jammer : np.ndarray
        Jammer signal
    signal : np.ndarray
        Original signal (reference for power)
    jnr_db : float
        Desired JNR in dB (positive = jammer stronger than signal)
    
    Returns
    -------
    np.ndarray
        Scaled jammer signal
    """
    sig_power = compute_signal_power(signal)
    jam_power = compute_signal_power(jammer)
    
    # JNR = P_jammer / P_signal
    jnr_linear = 10 ** (jnr_db / 10.0)
    
    # Scale factor
    scale = np.sqrt((jnr_linear * sig_power) / jam_power)
    
    return jammer * scale


# ---------------------------
# Main injection function
# ---------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Inject interference into I/Q binary files"
    )
    ### Parser arguments ###
    # I/O
    ap.add_argument("--input", required=True, help="Input I/Q file")
    ap.add_argument("--output", required=True, help="Output I/Q file")
    ap.add_argument("--fs", type=float, required=True, help="Sample rate (Hz)")
    ap.add_argument("--dtype", default="int16", choices=list(DTYPE_MAP), 
                    help="Data type (default: int16)")
    
    # Segmentation selection
    ap.add_argument("--start-seconds", type=float, default=0.0, help = "Start time (in seconds, default: 0)")
    ap.add_argument("--segment-seconds", type=float, default=2.0, help = "Segment duration (in seconds, default: 2.0)")
    ap.add_argument("--iq-order", default="IQ", choices=["IQ", "QI"])
    ap.add_argument("--num-segments", type=int, default=1, help="Number of segments to process (default: 1)")

    
    # Jammer type
    ap.add_argument("--jammer", required=True, 
                    choices=["cw", "pulse", "chirp", "barrage", "multitone", "swept"],
                    help="Jammer type")
    
    # Power
    ap.add_argument("--jnr", type=float, default=0.0,
                    help="Jammer-to-Noise Ratio in dB (default: 0 = same power as signal)")
    
    # CW parameters
    ap.add_argument("--freq-offset", type=float, default=0.0,
                    help="Frequency offset from baseband (Hz)")
    
    # Pulse parameters
    ap.add_argument("--pulse-width", type=float, default=100e-6,
                    help="Pulse width (seconds, default: 100 µs)")
    ap.add_argument("--pulse-period", type=float, default=1e-3,
                    help="Pulse period (seconds, default: 1 ms)")
    
    # Chirp parameters
    ap.add_argument("--chirp-bw", type=float, default=1e6,
                    help="Chirp bandwidth (Hz, default: 1 MHz)")
    ap.add_argument("--chirp-duration", type=float, default=1e-3,
                    help="Chirp duration (seconds, default: 1 ms)")
    
    # Multitone parameters
    ap.add_argument("--tones", type=str, default="100e3,200e3,300e3",
                    help="Comma-separated tone frequencies for multitone (Hz)")
    
    # Swept tone parameters
    ap.add_argument("--sweep-min", type=float, default=-500e3,
                    help="Sweep minimum frequency (Hz)")
    ap.add_argument("--sweep-max", type=float, default=500e3,
                    help="Sweep maximum frequency (Hz)")
    ap.add_argument("--sweep-rate", type=float, default=1e6,
                    help="Sweep rate (Hz/second)")
    
    # Other
    ap.add_argument("--seed", type=int, default=None,
                    help="Random seed for barrage jammer")
    
    args = ap.parse_args()
    
    path = Path(args.input)
    if not path.exists():
        raise FileNotFoundError(path)

    # Load original signal
    """Open interleaved I/Q file as 1D memmap of real values (I,Q,I,Q,...)."""
    if args.dtype not in DTYPE_MAP:
        raise ValueError(f"Unsupported dtype '{args.dtype}'. Choose from {list(DTYPE_MAP)}.")
    mm = np.memmap(path, mode='r', dtype=DTYPE_MAP[args.dtype])
    total_cplx = mm.size // 2  # Each complex sample has 2 real values
    seconds_total = total_cplx / args.fs
    print(f"[info] File: {path.name}")
    print(f"[info] dtype={args.dtype}, samples={total_cplx:,} complex, duration≈{seconds_total:.2f}s at fs={args.fs:g} Hz")

    # Segment sizing in complex samples
    start_cplx = int(round(args.start_seconds * args.fs))
    seg_cplx = int(round(args.segment_seconds * args.fs))
    if seg_cplx <= 0:
        raise ValueError("--segment-seconds must be > 0")
    if start_cplx >= total_cplx:
        print("[warn] Start exceeds file length; nothing to do.")
        return
    
    for segment_idx in range(args.num_segments):
        seg_start = start_cplx + segment_idx * seg_cplx
        if seg_start >= total_cplx:
            print(f"[warn] Segment {segment_idx} start exceeds file length; stopping.")
            break
        seg_end = min(seg_start + seg_cplx, total_cplx)
        
        signal = read_complex_chunk(mm, seg_start, seg_end, args.iq_order, args.dtype)

        if args.jammer == "chirp":
            jammer = generate_chirp_jammer(
            seg_cplx, args.fs, args.chirp_bw, 
            args.chirp_duration, args.freq_offset
        )
        print(f"[info] Chirp: BW={args.chirp_bw/1e6:.2f} MHz, "
              f"duration={args.chirp_duration*1e3:.2f} ms")
        
        jammer_scaled = scale_to_jnr(jammer, signal, args.jnr)

            # Add jammer to signal
        print(f"[info] Adding jammer to signal...")
        jammed_signal = signal + jammer_scaled
        
        # Save
        print(f"[info] Saving to {args.output + str(segment_idx) + '.bin'}...")
        save_iq_file(args.output + str(segment_idx) + '.bin', jammed_signal, args.dtype)
    
    print(f"[done] Jammed file created successfully!")
    print(f"[hint] Visualize with: python generate_spectrogram.py --input {args.output} --fs {args.fs}")


if __name__ == "__main__":
    main()
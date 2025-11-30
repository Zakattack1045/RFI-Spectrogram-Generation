#!/usr/bin/env python3
"""
Generate spectrogram images from large interleaved I/Q recordings without
loading the whole file into memory. You can request multiple smaller
spectrograms (segments) of custom duration.

MODIFIED: Now uses pspectrum-style smart defaults for cleaner spectrograms.

Examples
--------
# Auto-everything (like MATLAB pspectrum)
python generate_spectrogram.py \
  --input raw_iq.dat --fs 24e6 \
  --segment-seconds 1 --num-segments 1

# Two 2-second spectrograms (first 4 seconds total)
python generate_spectrogram.py \
  --input raw_iq.dat --fs 4e6 \
  --segment-seconds 2 --num-segments 2 \
  --output-template "spec_{index:03d}.png"

# With manual parameters
python generate_spectrogram.py \
  --input raw_iq.dat --fs 10e6 \
  --segment-seconds 1 --num-segments 10 \
  --start-seconds 30 \
  --n-fft 16384 --overlap-pct 80

Notes
-----
- Assumes interleaved I/Q samples (I0, Q0, I1, Q1, ...).
- Supports dtype int16, float32, int8 (typical SDR formats).
- Uses memmap to read only the requested chunk(s).
"""

import argparse
import math
from pathlib import Path

import numpy as np
import json
import random
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import uniform_filter, median_filter
from Jammers import generate_cw_tone, generate_chirp_jammer, generate_barrage_jammer, generate_pulsed_jammer
# from utils import SpectrogramDataset


# ---------------------------
# I/O and chunking utilities
# ---------------------------

DTYPE_MAP = {
    "int16": np.int16,
    "float32": np.float32,
    "int8": np.int8,
}

def bytes_per_complex(dtype: np.dtype) -> int:
    """Return bytes consumed by one complex sample (I+Q) for given real dtype."""
    return 2 * np.dtype(dtype).itemsize  # I and Q

def open_memmap_interleaved(path: str, dtype: str) -> np.memmap:
    """Open interleaved I/Q file as 1D memmap of real values (I,Q,I,Q,...)."""
    if dtype not in DTYPE_MAP:
        raise ValueError(f"Unsupported dtype '{dtype}'. Choose from {list(DTYPE_MAP)}.")
    return np.memmap(path, mode='r', dtype=DTYPE_MAP[dtype])

def total_complex_samples(mm: np.memmap) -> int:
    """Total number of complex samples (I,Q pairs) in the memmap."""
    return mm.size // 2

def read_complex_chunk(
    mm: np.memmap,
    start_complex: int,
    n_complex: int,
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
    end_real = start_real + 2 * n_complex
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


# ---------------------------
# STFT and plotting
# ---------------------------

def make_window(n_fft: int, kind: str = "hann") -> np.ndarray:
    if kind.lower() == "hann":
        # periodic Hann window (STFT conv)
        return np.hanning(n_fft).astype(np.float32)
    elif kind.lower() == "hamming":
        return np.hamming(n_fft).astype(np.float32)
    elif kind.lower() == "rect":
        return np.ones(n_fft, dtype=np.float32)
    else:
        raise ValueError("Unsupported window. Use 'hann', 'hamming', or 'rect'.")

def stft_complex_iq(x: np.ndarray, fs: float, n_fft: int, hop: int, window: np.ndarray):
    """
    Compute complex STFT (frequency last axis) using sliding window view.
    Returns (S, freqs, times):
      S: complex STFT [frames, n_fft]
      freqs: frequency vector [Hz] length n_fft (fftshifted to [-fs/2, fs/2))
      times: frame center times [s] length frames
    """
    if x.size < n_fft:
        return np.empty((0, n_fft), dtype=np.complex64), np.empty(n_fft), np.empty(0)

    # Strided windowing view; step by hop
    frames = sliding_window_view(x, n_fft)[::hop]
    # Apply window (broadcast multiply)
    frames = frames * window[None, :]
    # FFT with shift to center DC
    S = np.fft.fftshift(np.fft.fft(frames, n=n_fft, axis=-1), axes=-1)

    # Frequency axis (fftshifted)
    freqs = np.fft.fftshift(np.fft.fftfreq(n_fft, d=1.0/fs))  # Hz
    # Time stamps: center of each frame
    times = (np.arange(S.shape[0]) * hop + (n_fft // 2)) / fs

    return S.astype(np.complex64), freqs.astype(np.float32), times.astype(np.float32)

def power_db(S: np.ndarray, floor_db: float = -160.0) -> np.ndarray:
    """Convert complex STFT to power dB; clamp to a floor to avoid -inf."""
    P = (S.real ** 2 + S.imag ** 2).astype(np.float32)
    with np.errstate(divide='ignore'):
        db = 10.0 * np.log10(np.maximum(P, 1e-30))
    db = np.maximum(db, floor_db)
    return db

def jsr_scaling(signal: np.ndarray, jammer: np.ndarray, jsr_db: float) -> np.ndarray:
    power_sig = np.mean(np.abs(signal) ** 2)
    power_jam = np.mean(np.abs(jammer) ** 2)

    jsr_linear = 10 ** (jsr_db / 10.0)

    scale = np.sqrt((jsr_linear * power_sig) / power_jam)

    return jammer * scale

def save_spectrogram(
    S_db: np.ndarray,  # CHANGED: Now expects dB values directly, not normalized
    out_path: str,
    vmin: float = None,
    vmax: float = None,
    cmap: str = "jet",
    gamma: float = 1.0
):
    """
    Save spectrogram image (PNG) with axes in seconds and MHz.
    S_db shape: [frames, n_fft] in dB
    gamma: Gamma correction factor. Values < 1.0 boost mid-range values (increases contrast).
           Default 1.0 (no correction). Try 0.5-0.7 to make jammers more visible.
    """

    # Auto-detect reasonable dB range if not specified
    # CHANGED: More aggressive percentile range for better auto-detection
    if vmin is None or vmax is None:
        p1 = np.percentile(S_db, 1)  # CHANGED: 1st percentile instead of 5th
        p99 = np.percentile(S_db, 99)  # CHANGED: 99th percentile
        if vmin is None:
            vmin = p1 - 2  # CHANGED: Less padding below
        if vmax is None:
            vmax = p99 + 2  # CHANGED: Less padding above
        print(f"[info] Auto dB range: {vmin:.1f} to {vmax:.1f} dB")

    S_norm = np.clip((S_db - vmin) / (vmax - vmin), 0, 1)

    # Apply gamma correction to boost contrast (helps make jammers more visible)
    if gamma != 1.0:
        S_norm = np.power(S_norm, gamma)

    fig = plt.figure(frameon=False)
    fig.set_size_inches(S_norm.shape[1]/100, S_norm.shape[0]/100)

    ax = plt.Axes(fig, [0., 0., 1., 1.])  # Fill entire figure
    ax.set_axis_off()  # No axes
    fig.add_axes(ax)
    
    # 5. Plot with NO interpolation (exact pixels)
    ax.imshow(S_norm.T, aspect='auto', origin='lower', 
              cmap=cmap, interpolation='nearest')
    
    # 6. Save with NO padding
    plt.savefig(out_path, dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def save_binary_mask(jammer_db, out_path, percentile: float = 95.0):
    threshold = np.percentile(jammer_db, percentile)
    mask = (jammer_db > threshold).astype(np.uint8)
    fig = plt.figure(frameon=False)
    fig.set_size_inches(mask.shape[1]/100, mask.shape[0]/100)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    
    # Binary colormap: 0=black, 1=white
    ax.imshow(mask.T, aspect='auto', origin='lower', 
              cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    
    plt.savefig(out_path, dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

# CHANGED: New function to compute smart defaults like pspectrum
def compute_smart_defaults(fs: float, seg_samples: int):
    """
    Compute smart STFT parameters like MATLAB's pspectrum.
    
    pspectrum targets ~100-200 ms windows for good time-frequency balance.
    """
    # Target window duration (100 ms is a good default)
    target_window_s = 0.1
    
    # Compute power-of-2 FFT size close to target
    target_samples = fs * target_window_s
    n_fft = int(2 ** np.ceil(np.log2(target_samples)))
    
    # Don't make window too large relative to segment
    n_fft = min(n_fft, seg_samples // 8)
    
    # Minimum FFT size
    n_fft = max(n_fft, 512)
    
    # 80% overlap (pspectrum default)
    hop = int(0.2 * n_fft)
    
    return n_fft, hop


def generate_spectrogram(
        x: np.ndarray, jammer: np.ndarray,
        fs: float, n_fft: int,
        hop: int, window: np.ndarray,
        out_path: str, jammer_out_path: str,
        filter: tuple,
        vmin: float = None, vmax: float = None,
        gamma: float = 1.0):
    # STFT
    J, freqs, times = stft_complex_iq(jammer, fs, n_fft, hop, window)
    S, freqs, times = stft_complex_iq(x, fs, n_fft, hop, window)
    if S.size == 0 or J.size == 0:
        print(f"[warn] No frames for segment. Skipping.")
        return

    # Option 1: Correct power addition in linear scale
    # Step 1: Compute power in LINEAR scale
    S_power = (S.real ** 2 + S.imag ** 2).astype(np.float32)
    J_power = (J.real ** 2 + J.imag ** 2).astype(np.float32)

    # Step 2: Convert clean signal to dB for filtering
    with np.errstate(divide='ignore'):
        S_db = 10.0 * np.log10(np.maximum(S_power, 1e-30))
    S_db = np.maximum(S_db, -160.0)  # Floor to avoid -inf

    # Step 3: Apply median filter to clean signal in dB space (reduces noise)
    S_db_filtered = median_filter(S_db, size=(filter[0], filter[1]))

    # Step 4: Convert filtered clean signal back to LINEAR power
    S_power_filtered = 10 ** (S_db_filtered / 10.0)

    # Step 5: Add LINEAR powers (this is the CORRECT way to combine signals!)
    Total_power = S_power_filtered + J_power

    # Step 6: Convert combined power to dB
    with np.errstate(divide='ignore'):
        Out_db = 10.0 * np.log10(np.maximum(Total_power, 1e-30))
    Out_db = np.maximum(Out_db, -160.0)

    # Step 7: Optional light smoothing on combined spectrogram
    # Out_db = median_filter(Out_db, size=(3, 3))

    # Save jammer ground truth (convert jammer power to dB)
    with np.errstate(divide='ignore'):
        J_db = 10.0 * np.log10(np.maximum(J_power, 1e-30))
    J_db = np.maximum(J_db, -160.0)

    save_binary_mask(J_db, jammer_out_path)
    save_spectrogram(Out_db, out_path, vmin=vmin, vmax=vmax, gamma=gamma)

    png_size = Path(out_path).stat().st_size / (1024 * 1024)
    print(f"[ok] {out_path}  ({png_size:.2f} MB, dB: {Out_db.min():.1f} to {Out_db.max():.1f})")




# ---------------------------
# Main segmenter
# ---------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Generate spectrograms with pspectrum-style smart defaults."
    )
    ap.add_argument("--input", required=True, help="Path to interleaved I/Q binary file")
    ap.add_argument("--fs", type=float, required=True, help="Sample rate in Hz (e.g., 24e6)")
    ap.add_argument("--dtype", default="float32", choices=list(DTYPE_MAP), 
                    help="Real sample dtype (default float32)")  # CHANGED: default to float32
    ap.add_argument("--iq-order", default="IQ", choices=["IQ", "QI"], help="Interleave order (default IQ)")
    ap.add_argument("--decimate", type=int, default=1, help="Optional integer decimation factor (default 1)")

    # STFT controls
    # CHANGED: n_fft and hop now optional with smart defaults
    ap.add_argument("--n-fft", type=int, default=16384,
                    help="FFT length (default: auto, ~100ms window)")
    ap.add_argument("--hop", type=int, default=None,
                    help="Hop size in samples (default: auto, 80%% overlap)")
    # CHANGED: Added overlap-pct for easier control
    ap.add_argument("--overlap-pct", type=float, default=80,
                    help="Overlap percentage if --hop not specified (default: 80)")
    ap.add_argument("--window", default="hann", choices=["hann", "hamming", "rect"],
                    help="Window type (default hann)")  # CHANGED: default to hann

    # Chunking controls
    ap.add_argument("--segment-seconds", type=float, default=1.0,
                    help="Duration per spectrogram (seconds, default 1.0)")  # CHANGED: default to 1.0
    ap.add_argument("--num-segments", type=int, default=1, help="How many spectrograms to generate")
    ap.add_argument("--start-seconds", type=float, default=0.0, help="Start offset (seconds) in the file")

    # CHANGED: Added median filter controls
    ap.add_argument("--median-filter", type=int, default=15,
                    help="Median filter size for noise reduction (0=off, default: 5)")
    ap.add_argument("--median-freq", type=int, default=3,
                    help="Median filter size in frequency direction (default: 3)")

    # Output
    ap.add_argument("--output-dir", default="output_spectrograms",
                    help="Directory to save spectrogram images and metadata")
    ap.add_argument("--title", default="Spectrogram", help="Title prefix for plots")
    ap.add_argument("--save-numpy", action="store_true",
                    help="Save spectrograms as compressed numpy arrays (.npz) in addition to PNG")
    
    # CHANGED: Simplified to just vmin/vmax (removed normalization methods)
    ap.add_argument("--vmin", type=float, default=None,
                    help="Min dB for colorbar (default: auto)")
    ap.add_argument("--vmax", type=float, default=None,
                    help="Max dB for colorbar (default: auto)")
    ap.add_argument("--gamma", type=float, default=1.0,
                    help="Gamma correction factor for visibility (default: 1.0, try 0.5-0.7 to boost jammer contrast)")

    # Jammer Control
    ap.add_argument("--jsr-lb", type=float, default=0.0, help="Jammer-to-noise ratio lower bound in dB")
    ap.add_argument("--jsr-ub", type=float, default=0.0, help="Jammer-to-noise ratio upper bound in dB")
    ap.add_argument("--jammer-type", type=str, choices=["chirp", "none"], default="none",)
    ap.add_argument("--chirp-bw", type=float, default=1e6, help="Chirp bandwidth in Hz (default 1e6)")
    ap.add_argument("--chirp-duration", type=float, default=0.25, help="Chirp duration in seconds (default 0.1)")
    args = ap.parse_args()

    path = Path(args.input)
    if not path.exists():
        raise FileNotFoundError(path)

    # Open as memmap of real interleaved samples
    mm = open_memmap_interleaved(str(path), args.dtype)
    total_cplx = total_complex_samples(mm)
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

    # CHANGED: Compute smart defaults if not provided
    if args.n_fft is None or args.hop is None:
        default_nfft, default_hop = compute_smart_defaults(args.fs, seg_cplx)
        n_fft = args.n_fft or default_nfft
        hop = args.hop or int(n_fft * (1 - args.overlap_pct/100))
        print(f"[info] Using smart defaults: n_fft={n_fft}, hop={hop} ({args.overlap_pct:.0f}% overlap)")
    else:
        n_fft = args.n_fft
        hop = args.hop

    jammers = ['CW', 'pulse','chirp', 'barrage']

    # STFT window
    win = make_window(n_fft, args.window)

    # Effective fs if decimated
    eff_fs = args.fs / max(1, args.decimate)
    
    # CHANGED: Better info display
    freq_res = eff_fs / n_fft
    time_res = hop / eff_fs
    overlap_actual = 100 * (1 - hop / n_fft)
    print(f"\n{'='*60}")
    print(f"STFT Configuration (pspectrum-style)")
    print(f"{'='*60}")
    print(f"Sample rate:        {eff_fs/1e6:.1f} MHz")
    print(f"Segment duration:   {args.segment_seconds:.3f} s ({seg_cplx:,} samples)")
    print(f"FFT size:           {n_fft}")
    print(f"Hop size:           {hop} ({overlap_actual:.1f}% overlap)")
    print(f"Window:             {args.window}")
    print(f"Frequency res:      {freq_res:.2f} Hz ({freq_res/1e3:.3f} kHz)")
    print(f"Time res:           {time_res*1e3:.3f} ms")
    if args.median_filter > 0:
        print(f"Median filter:      ({args.median_filter}, {args.median_freq})")
    print(f"{'='*60}\n")

    if args.decimate != 1:
        print(f"[info] Decimation: /{args.decimate}  -> effective fs={eff_fs:g} Hz")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "spectrograms").mkdir(exist_ok=True)
    (output_dir / "masks").mkdir(exist_ok=True)
    if args.save_numpy:
        (output_dir / "spectrograms_numpy").mkdir(exist_ok=True)
        (output_dir / "masks_numpy").mkdir(exist_ok=True)

    # Initialize metadata
    metadata = {
        "fs": args.fs,
        "n_fft": n_fft,
        "hop": hop,
        "segment_seconds": args.segment_seconds,
        "samples": []
    }

    # Loop over segments
    made = 0
    for idx in range(args.num_segments):
        jsr = random.random() * (args.jsr_ub - args.jsr_lb) + args.jsr_lb
        freq_offset = random.randint(-int(args.fs/4), int(args.fs/4))
        jammer_BW = random.randint(int(args.fs/24), int(args.fs/8))
        name = f"sample_{idx:05d}_jsr{jsr:+.1f}"
        seg_start = start_cplx + idx * seg_cplx
        if seg_start >= total_cplx:
            print(f"[info] Reached end of file at segment {idx}.")
            break
        seg_end = min(seg_start + seg_cplx, total_cplx)
        need = seg_end - seg_start
        if need < max(n_fft, hop):
            print(f"[warn] Segment {idx} too short for STFT (need at least n_fft={n_fft}). Skipping.")
            continue

        # Read chunk
        x = read_complex_chunk(
            mm, seg_start, need,
            iq_order=args.iq_order, dtype=args.dtype, scale_to_float=True
        )

        # Add interference
        match jammers[made%len(jammers)]:
            case 'CW':
                bin_width = eff_fs / n_fft
                k = int(np.round(freq_offset / bin_width))
                freq_offset = k * bin_width
                jammer = generate_cw_tone(
                    n_samples=need,
                    fs=args.fs,
                    freq_offset=freq_offset,
                    phase=0.0,
                    bandwidth=jammer_BW/8
                )
            case 'pulse':
                jammer = generate_pulsed_jammer(
                    n_samples=need,
                    fs=args.fs,
                    pulse_width=args.segment_seconds / 10,
                    pulse_period=args.segment_seconds / 4,
                    freq_offset=freq_offset,
                    phase=0.0,
                    bandwidth=jammer_BW/8
                )
            case 'chirp':
                jammer = generate_chirp_jammer(
                    n_samples=need,
                    fs=args.fs,
                    chirp_bw=jammer_BW,
                    chirp_duration=args.segment_seconds,
                    center_freq= random.randint(-int(args.fs/6), int(args.fs/6)),
                    bandwidth=jammer_BW/8
                )
            case "barrage":
                jammer = generate_barrage_jammer(
                    n_samples=need,
                    fs=args.fs
                )

        jammer = jsr_scaling(x, jammer, jsr)

        # Decimate jammer for mask generation to match final sample rate
        jammer_for_mask = jammer[::args.decimate] if args.decimate > 1 else jammer

        # Generate jammer mask
        mask_png_path = args.output_dir + f"/masks/{name}.png"
        x += jammer

        # Optional decimation (simple keep-every-N)
        if args.decimate > 1:
            x = x[::args.decimate]

        # Generate combined spectrogram
        spec_png_path = args.output_dir + f"/spectrograms/{name}.png"
        generate_spectrogram(
            x, jammer, eff_fs, n_fft, hop, win,
            spec_png_path, mask_png_path, filter=(args.median_filter, args.median_freq),
            vmin=args.vmin, vmax=args.vmax, gamma=args.gamma
        )
        
        
        # Add each sample
        sample_metadata = {
            "id": idx,
            "jammer_type": jammers[idx%len(jammers)],
            "jsr_values": jsr,
            "frequency_offset": freq_offset,
            "jammer_bw": jammer_BW,
            "spectrogram": f"spectrograms/sample_{idx:05d}_jsr{jsr:+.1f}.png",
            "mask": f"masks/sample_{idx:05d}_jsr{jsr:+.1f}.png",
            "time_start": float(start_cplx / args.fs),
        }
        if args.save_numpy:
            sample_metadata["spectrogram_numpy"] = f"spectrograms_numpy/sample_{idx:05d}_jsr{jsr:+.1f}.npz"
            sample_metadata["mask_numpy"] = f"masks_numpy/sample_{idx:05d}_jsr{jsr:+.1f}.npz"
        metadata["samples"].append(sample_metadata)

        made += 1

    if made == 0:
        print("[info] No spectrograms generated.")
    else:
        # Save metadata
        with open(args.output_dir + "/metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"\n[done] Generated {made} spectrogram(s).")

    # dataset = SpectrogramDataset(args.output_dir,height=64, width=64)
    # for i in range(len(dataset)):
    #     # Load spectrogram
    #     spectrogram, true_mask_2d = dataset[i]
    #     spectrogram = spectrogram.squeeze().numpy()  # (H, W)
    #     true_mask_2d = true_mask_2d.squeeze().numpy()  # (H, W)

    #     # Create comparison figure
    #     fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    #     # Original spectrogram
    #     axes[0].imshow(spectrogram, cmap='viridis')
    #     axes[0].set_title('Input Spectrogram')
    #     axes[0].axis('off')

    #     # Ground truth mask
    #     axes[1].imshow(true_mask_2d, cmap='gray', vmin=0, vmax=1)
    #     axes[1].set_title('Ground Truth Mask')
    #     axes[1].axis('off')

    #     plt.tight_layout()
    #     plt.savefig(args.output_dir +f'/sample_{i:03d}.png', dpi=150)
    #     plt.close()


    


if __name__ == "__main__":
    main()
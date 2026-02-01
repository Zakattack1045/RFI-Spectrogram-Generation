import argparse
import math
from pathlib import Path

import numpy as np
import json
import random
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import uniform_filter, median_filter

import utils

DTYPE_MAP = {
    "int16": np.int16,
    "float32": np.float32,
    "int8": np.int8,
}

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
    ap.add_argument("--vmin", type=float, default=None,
                    help="Min dB for colorbar (default: auto)")
    ap.add_argument("--vmax", type=float, default=None,
                    help="Max dB for colorbar (default: auto)")
    ap.add_argument("--gamma", type=float, default=1.0,
                    help="Gamma correction factor for visibility (default: 1.0, try 0.5-0.7 to boost jammer contrast)")
    # Output
    ap.add_argument("--mask" , default=False, help="Toggle for binary mask output")
    ap.add_argument("--output-dir", default="output_spectrograms",
                    help="Directory to save spectrogram images and metadata")
    ap.add_argument("--title", default="Spectrogram", help="Title prefix for plots")
    
    args = ap.parse_args()

    path = Path(args.input)
    if not path.exists():
        raise FileNotFoundError(path)

    # Open as memmap of real interleaved samples
    mm = utils.open_memmap_interleaved(str(path), args.dtype)
    total_cplx = utils.total_complex_samples(mm)
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
        default_nfft, default_hop = utils.compute_smart_defaults(args.fs, seg_cplx)
        n_fft = args.n_fft or default_nfft
        hop = args.hop or int(n_fft * (1 - args.overlap_pct/100))
        print(f"[info] Using smart defaults: n_fft={n_fft}, hop={hop} ({args.overlap_pct:.0f}% overlap)")
    else:
        n_fft = args.n_fft
        hop = args.hop



    # STFT window
    win = utils.make_window(n_fft, args.window)

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
        name = f"sample_{idx:05d}"
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
        x = utils.read_complex_chunk(
            mm, seg_start, need,
            iq_order=args.iq_order, dtype=args.dtype, scale_to_float=True
        )

        # Optional decimation (simple keep-every-N)
        if args.decimate > 1:
            x = x[::args.decimate]

        # Generate combined spectrogram
        spec_png_path = args.output_dir + f"/spectrograms/{name}.png"

        S, freqs, times = utils.stft_complex_iq(x, eff_fs, n_fft, hop, win)

        S_power = (S.real ** 2 + S.imag ** 2).astype(np.float32)

        with np.errstate(divide='ignore'):
            S_db = 10.0 * np.log10(np.maximum(S_power, 1e-30))
        S_db = np.maximum(S_db, -160.0)  # Floor to avoid -inf

        S_db_filtered = median_filter(S_db, size=(args.median_filter, args.median_freq))
        S_power_filtered = 10 ** (S_db_filtered / 10.0)

        with np.errstate(divide='ignore'):
            Out_db = 10.0 * np.log10(np.maximum(S_power_filtered, 1e-30))
        Out_db = np.maximum(Out_db, -160.0)

        if args.mask:
            utils.save_binary_mask(Out_db, spec_png_path)
        else:
            utils.save_spectrogram(Out_db, spec_png_path, vmin=args.vmin, vmax=args.vmax, gamma=args.gamma)
        
        
        # Add each sample
        sample_metadata = {
            "id": idx,
            "spectrogram": f"spectrograms/sample_{idx:05d}.png",
            "time_start": float(start_cplx / args.fs),
        }
        metadata["samples"].append(sample_metadata)

        made += 1

    if made == 0:
        print("[info] No spectrograms generated.")
    else:
        # Save metadata
        with open(args.output_dir + "/metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"\n[done] Generated {made} spectrogram(s).")

if __name__ == "__main__":
    main()
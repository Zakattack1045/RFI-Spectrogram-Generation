#!/usr/bin/env python3
"""
create_dataset.py - Combine multiple RFI detection datasets into a unified training set.

Creates a combined dataset from:
  - Clean signal spectrograms (cycled/repeated as needed)
  - IoT interference spectrograms (kept as sequential block)
  - Narrowband interference spectrograms (distributed among RFI slots)

Output is compatible with SpectrogramDataset class (spectrograms/ and masks/ subdirectories).

Usage:
    python create_dataset.py --total-samples 2000 --clean-duty-cycle 0.7
    python create_dataset.py --help
"""

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from PIL import Image
from tqdm import tqdm


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class DatasetConfig:
    """Configuration for dataset creation."""
    # Output settings
    total_samples: int = 2000
    output_dir: str = "Data/OUTPUTS/Combined_Dataset"
    image_size: Tuple[int, int] = (64, 64)

    # Duty cycle (clean vs RFI ratio)
    clean_duty_cycle: float = 0.50  

    # Input source paths
    clean_spec_dir: str = "Data/OUTPUTS/Clean_Specs_Out"
    iot_spec_dir: str = "Data/OUTPUTS/IoT_Specs_Out"
    nb_spec_dir: str = "Data/OUTPUTS/Train_NB"

    # Placement options
    shuffle_nb: bool = True
    random_seed: int = 42


# ============================================================================
# Data Source Classes
# ============================================================================

class DataSource:
    """
    Base class for loading data from a source directory.

    Expects directory structure:
        base_dir/
            spectrograms/  - Spectrogram images
            masks/         - Corresponding mask images
    """

    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.spec_dir = self.base_dir / "spectrograms"
        self.mask_dir = self.base_dir / "masks"
        self._load_files()

    def _load_files(self):
        """Discover available spectrogram/mask pairs."""
        if not self.spec_dir.exists():
            raise FileNotFoundError(f"Spectrograms directory not found: {self.spec_dir}")
        if not self.mask_dir.exists():
            raise FileNotFoundError(f"Masks directory not found: {self.mask_dir}")

        # Get all image files
        self.spec_files = sorted(self.spec_dir.glob("*.png"))
        self.mask_files = sorted(self.mask_dir.glob("*.png"))

        # Create mapping from stem (filename without extension) to full path
        self.spec_map = {f.stem: f for f in self.spec_files}
        self.mask_map = {f.stem: f for f in self.mask_files}

        # Find matching pairs (files that exist in both directories)
        self.available_samples = sorted(
            set(self.spec_map.keys()) & set(self.mask_map.keys())
        )

        if not self.available_samples:
            raise ValueError(f"No matching spectrogram/mask pairs found in {self.base_dir}")

    def __len__(self) -> int:
        return len(self.available_samples)

    def get_pair(self, index: int) -> Tuple[Path, Path]:
        """
        Return (spectrogram_path, mask_path) for given index.

        Args:
            index: Sample index (0 to len-1)

        Returns:
            Tuple of (spectrogram_path, mask_path)
        """
        if index < 0 or index >= len(self.available_samples):
            raise IndexError(f"Index {index} out of range [0, {len(self)})")

        sample_name = self.available_samples[index]
        return (self.spec_map[sample_name], self.mask_map[sample_name])


class CyclicDataSource(DataSource):
    """
    Data source that cycles when index exceeds available samples.

    Useful for clean images where we have limited samples (e.g., 126)
    but need to fill many more slots.
    """

    def get_pair(self, index: int) -> Tuple[Path, Path]:
        """Return pair with cyclic indexing (wraps around)."""
        cyclic_index = index % len(self.available_samples)
        sample_name = self.available_samples[cyclic_index]
        return (self.spec_map[sample_name], self.mask_map[sample_name])


# ============================================================================
# Image Processing
# ============================================================================

def load_and_resize(image_path: Path, size: Tuple[int, int], is_mask: bool = False) -> Image.Image:
    """
    Load image and resize to target size.

    Args:
        image_path: Path to the image file
        size: Target size (width, height)
        is_mask: If True, use NEAREST interpolation to preserve binary values

    Returns:
        Resized PIL Image
    """
    img = Image.open(image_path)

    # Use appropriate interpolation
    if is_mask:
        # NEAREST preserves binary 0/1 values in masks
        resample = Image.Resampling.NEAREST
    else:
        # BILINEAR for smooth spectrogram resizing
        resample = Image.Resampling.BILINEAR

    return img.resize(size, resample=resample)


def save_image(image: Image.Image, output_path: Path) -> None:
    """Save image, creating parent directories if needed."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


# ============================================================================
# Placement Algorithm
# ============================================================================

def create_interleaved_schedule(
    config: DatasetConfig,
    clean_source: DataSource,
    iot_source: DataSource,
    nb_source: DataSource
) -> List[Tuple[int, str, int]]:
    """
    Create the output schedule determining which source provides each sample.

    The algorithm:
    1. Calculate how many clean vs RFI samples based on duty cycle
    2. Create RFI "events": IoT as one contiguous block, NB as individual samples
    3. Randomly place the IoT block among the NB events
    4. Distribute clean samples with RANDOM durations between RFI events
       (no predictable pattern - clean periods vary in length)

    Args:
        config: Dataset configuration
        clean_source: Clean data source
        iot_source: IoT data source
        nb_source: Narrowband data source

    Returns:
        List of (output_index, source_type, source_index) tuples
        where source_type is 'clean', 'iot', or 'nb'
    """
    random.seed(config.random_seed)

    total = config.total_samples
    clean_count = int(total * config.clean_duty_cycle)
    rfi_count = total - clean_count

    # Check available RFI samples
    iot_available = len(iot_source)
    nb_available = len(nb_source)
    total_rfi_available = iot_available + nb_available

    if rfi_count > total_rfi_available:
        print(f"WARNING: Requested {rfi_count} RFI samples but only {total_rfi_available} available.")
        print(f"         Adjusting RFI count to {total_rfi_available}")
        rfi_count = total_rfi_available
        clean_count = total - rfi_count

    # Allocate RFI images: prioritize using all IoT (as one block), then NB
    iot_count = min(iot_available, rfi_count)
    nb_count = min(nb_available, rfi_count - iot_count)

    # -------------------------------------------------------------------------
    # Create RFI events list
    # IoT block = one event containing all IoT images (contiguous)
    # NB samples = individual events (one sample each)
    # -------------------------------------------------------------------------
    rfi_events = []

    # Add individual NB events
    nb_indices = list(range(nb_count))
    if config.shuffle_nb:
        random.shuffle(nb_indices)

    for nb_idx in nb_indices:
        rfi_events.append(('nb', nb_idx, 1))  # (type, source_index, length)

    # Insert IoT block at a random position among the NB events
    if iot_count > 0:
        # Random position: could be at start, middle, or end
        iot_insert_pos = random.randint(0, len(rfi_events))
        rfi_events.insert(iot_insert_pos, ('iot_block', 0, iot_count))

    # -------------------------------------------------------------------------
    # Distribute clean samples randomly among gaps
    # Gaps: before first event, between events, after last event
    # -------------------------------------------------------------------------
    num_gaps = len(rfi_events) + 1

    if num_gaps > 0 and clean_count > 0:
        # Generate random weights for each gap
        # Use exponential distribution for more natural variation
        weights = [random.expovariate(1.0) for _ in range(num_gaps)]
        total_weight = sum(weights)

        # Distribute clean samples proportionally to weights
        clean_per_gap = []
        remaining_clean = clean_count

        for i, w in enumerate(weights[:-1]):
            # Allocate proportionally, but ensure integer counts
            gap_clean = int(clean_count * w / total_weight)
            clean_per_gap.append(gap_clean)
            remaining_clean -= gap_clean

        # Last gap gets whatever remains (handles rounding)
        clean_per_gap.append(remaining_clean)
    else:
        clean_per_gap = [clean_count] if num_gaps > 0 else []

    # -------------------------------------------------------------------------
    # Build the final schedule
    # -------------------------------------------------------------------------
    schedule = []
    output_idx = 0
    clean_idx = 0
    iot_idx = 0

    for gap_idx, event in enumerate(rfi_events):
        # Add clean samples for this gap (random duration)
        for _ in range(clean_per_gap[gap_idx]):
            schedule.append((output_idx, 'clean', clean_idx))
            output_idx += 1
            clean_idx += 1

        # Add the RFI event
        event_type, event_start_idx, event_length = event

        if event_type == 'iot_block':
            # Add all IoT images contiguously
            for _ in range(event_length):
                schedule.append((output_idx, 'iot', iot_idx))
                output_idx += 1
                iot_idx += 1
        else:  # 'nb'
            schedule.append((output_idx, 'nb', event_start_idx))
            output_idx += 1

    # Add final gap of clean samples
    if clean_per_gap:
        for _ in range(clean_per_gap[-1]):
            schedule.append((output_idx, 'clean', clean_idx))
            output_idx += 1
            clean_idx += 1

    return schedule


# ============================================================================
# Main Dataset Creation
# ============================================================================

def validate_sources(config: DatasetConfig) -> None:
    """Validate that all source directories exist."""
    errors = []

    for name, path in [
        ("Clean", config.clean_spec_dir),
        ("IoT", config.iot_spec_dir),
        ("Narrowband", config.nb_spec_dir)
    ]:
        full_path = Path(path)
        if not full_path.exists():
            errors.append(f"{name} directory not found: {path}")
        elif not (full_path / "spectrograms").exists():
            errors.append(f"{name}/spectrograms subdirectory not found: {path}")
        elif not (full_path / "masks").exists():
            errors.append(f"{name}/masks subdirectory not found: {path}")

    if errors:
        raise FileNotFoundError("\n".join(errors))


def create_dataset(config: DatasetConfig) -> None:
    """
    Main function to create the combined dataset.

    Args:
        config: Dataset configuration
    """
    print("=" * 60)
    print("RFI Detection Dataset Creator")
    print("=" * 60)

    # Validate source directories
    validate_sources(config)

    # Initialize data sources
    clean_source = CyclicDataSource(config.clean_spec_dir)
    iot_source = DataSource(config.iot_spec_dir)
    nb_source = DataSource(config.nb_spec_dir)

    print(f"\nInput Sources:")
    print(f"  Clean:      {len(clean_source)} samples (will cycle as needed)")
    print(f"  IoT:        {len(iot_source)} samples (sequential block)")
    print(f"  Narrowband: {len(nb_source)} samples (distributed)")

    # Create schedule
    schedule = create_interleaved_schedule(config, clean_source, iot_source, nb_source)

    # Calculate statistics
    type_counts = {'clean': 0, 'iot': 0, 'nb': 0}
    for _, source_type, _ in schedule:
        type_counts[source_type] += 1

    actual_clean_pct = type_counts['clean'] / len(schedule) * 100
    actual_rfi_pct = (type_counts['iot'] + type_counts['nb']) / len(schedule) * 100

    print(f"\nOutput Configuration:")
    print(f"  Total samples:     {len(schedule)}")
    print(f"  Target duty cycle: {config.clean_duty_cycle:.1%} clean / {1-config.clean_duty_cycle:.1%} RFI")
    print(f"  Actual allocation: {type_counts['clean']} clean ({actual_clean_pct:.1f}%), "
          f"{type_counts['iot']} IoT, {type_counts['nb']} NB ({actual_rfi_pct:.1f}% RFI)")
    print(f"  Output directory:  {config.output_dir}")
    print(f"  Image size:        {config.image_size[0]}x{config.image_size[1]}")

    # Create output directories
    output_path = Path(config.output_dir)
    spec_out_dir = output_path / "spectrograms"
    mask_out_dir = output_path / "masks"
    spec_out_dir.mkdir(parents=True, exist_ok=True)
    mask_out_dir.mkdir(parents=True, exist_ok=True)

    # Map source types to data sources
    sources = {
        'clean': clean_source,
        'iot': iot_source,
        'nb': nb_source
    }

    # Process images with progress bar
    print("\nProcessing images...")
    for output_idx, source_type, source_idx in tqdm(schedule, desc="Creating dataset"):
        source = sources[source_type]
        spec_path, mask_path = source.get_pair(source_idx)

        # Load, resize, and save spectrogram
        spec_img = load_and_resize(spec_path, config.image_size, is_mask=False)
        spec_out = spec_out_dir / f"sample_{output_idx:05d}.png"
        save_image(spec_img, spec_out)

        # Load, resize, and save mask
        mask_img = load_and_resize(mask_path, config.image_size, is_mask=True)
        mask_out = mask_out_dir / f"sample_{output_idx:05d}.png"
        save_image(mask_img, mask_out)

    # Save metadata for reproducibility
    metadata = {
        "total_samples": len(schedule),
        "clean_duty_cycle": config.clean_duty_cycle,
        "image_size": list(config.image_size),
        "sources": {
            "clean": str(config.clean_spec_dir),
            "iot": str(config.iot_spec_dir),
            "nb": str(config.nb_spec_dir)
        },
        "counts": type_counts,
        "random_seed": config.random_seed,
        "shuffle_nb": config.shuffle_nb
    }

    metadata_path = output_path / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"DONE: Created {len(schedule)} samples in {config.output_dir}")
    print(f"      Metadata saved to {metadata_path}")
    print(f"{'=' * 60}")


# ============================================================================
# CLI Interface
# ============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Combine RFI detection datasets into a unified training set.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default configuration (70%% clean, 30%% RFI, 2000 samples)
  python create_dataset.py

  # Custom configuration
  python create_dataset.py --total-samples 5000 --clean-duty-cycle 0.6

  # Specify custom output directory
  python create_dataset.py --output-dir Data/OUTPUTS/Train_Combined

  # Reproducible with specific seed, no NB shuffling
  python create_dataset.py --seed 123 --no-shuffle-nb
        """
    )

    # Output settings
    parser.add_argument("--total-samples", type=int, default=2000,
                        help="Total number of output samples (default: 2000)")
    parser.add_argument("--clean-duty-cycle", type=float, default=0.50,
                        help="Fraction of clean samples, 0.0-1.0 (default: 0.50)")
    parser.add_argument("--output-dir", type=str, default="Data/OUTPUTS/Combined_Dataset",
                        help="Output directory path (default: Data/OUTPUTS/Combined_Dataset)")
    parser.add_argument("--image-size", type=int, default=64,
                        help="Output image size, square (default: 64)")

    # Placement options
    parser.add_argument("--shuffle-nb", action="store_true", default=True,
                        help="Shuffle narrowband placement among RFI slots (default: True)")
    parser.add_argument("--no-shuffle-nb", action="store_false", dest="shuffle_nb",
                        help="Don't shuffle narrowband placement")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")

    # Source directories
    parser.add_argument("--clean-dir", type=str, default="Data/OUTPUTS/Clean_Specs_Out",
                        help="Clean spectrograms directory")
    parser.add_argument("--iot-dir", type=str, default="Data/OUTPUTS/IoT_Specs_Out",
                        help="IoT spectrograms directory")
    parser.add_argument("--nb-dir", type=str, default="Data/OUTPUTS/Train_NB",
                        help="Narrowband spectrograms directory")

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    config = DatasetConfig(
        total_samples=args.total_samples,
        output_dir=args.output_dir,
        image_size=(args.image_size, args.image_size),
        clean_duty_cycle=args.clean_duty_cycle,
        clean_spec_dir=args.clean_dir,
        iot_spec_dir=args.iot_dir,
        nb_spec_dir=args.nb_dir,
        shuffle_nb=args.shuffle_nb,
        random_seed=args.seed
    )

    try:
        create_dataset(config)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return 1
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        return 1
    except Exception as e:
        print(f"ERROR: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

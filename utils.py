import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image

# Lava imports for inference processes
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.process.variable import Var
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.resources import CPU
from lava.magma.core.model.sub.model import AbstractSubProcessModel

try:
    from lava.proc import embedded_io as eio
    from lava.magma.core.resources import Loihi2NeuroCore
    LOIHI2_AVAILABLE = True
except ImportError:
    LOIHI2_AVAILABLE = False

class SpectrogramDataset(Dataset):
    """
    Dataset for RFI detection training.
    Expected directory structure:
        dir/
            spectrograms/  - Spectrograms with interference + signal
            masks/        - Binary ground truth masks (white=interference, black=clean)
    Files should have matching names across both directories.
    """
    def __init__(self, dir, transform=None, target_transform=None, height=256, width=256):
        self.spectrogram_dir = os.path.join(dir, "spectrograms")
        self.mask_dir = os.path.join(dir, "masks")

        # Get list of files
        if os.path.exists(self.spectrogram_dir):
            self.file_list = sorted([f for f in os.listdir(self.spectrogram_dir)
                                    if f.endswith(('.png', '.jpg', '.jpeg'))])
        else:
            self.file_list = []

        self.img_tf = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((height, width), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        self.mask_size = (height, width)
        self.nearest = transforms.InterpolationMode.NEAREST

    def __len__(self):
        return len(self.file_list)

    def _load_binary_mask(self, path):
        mask = Image.open(path).convert('L')
        mask = transforms.functional.resize(mask, self.mask_size, interpolation=self.nearest)
        mask = np.array(mask, dtype=np.uint8)
        # Normalize to 0 or 1
        if (mask.max() > 1):
            mask = (mask > 127).astype(np.uint8)
        mask = torch.from_numpy(mask).float().unsqueeze(0)
        return mask

    def __getitem__(self, idx):
        filename = self.file_list[idx]

        # Load spectrogram (interference + signal)
        spec_path = os.path.join(self.spectrogram_dir, filename)
        spectrogram = self.img_tf(Image.open(spec_path))  # Returns (C, H, W) = (1, 32, 32)

        # Load ground truth binary mask
        mask_path = os.path.join(self.mask_dir, filename)
        mask = self._load_binary_mask(mask_path)  # Returns (C, H, W) = (1, 32, 32)

        return spectrogram, mask


# ============================================================================
# Adapter Processes for connecting CPU and Loihi2 hardware
# ============================================================================

class InputAdapter(AbstractProcess):
    """
    Input adapter process to transfer data between CPU and network.
    """
    def __init__(self, shape):
        super().__init__(shape=shape)
        self.inp = InPort(shape=shape)
        self.out = OutPort(shape=shape)


class OutputAdapter(AbstractProcess):
    """
    Output adapter process to transfer data from network to CPU.
    """
    def __init__(self, shape):
        super().__init__(shape=shape)
        self.inp = InPort(shape=shape)
        self.out = OutPort(shape=shape)


# CPU ProcessModels for Adapters
@implements(proc=InputAdapter, protocol=LoihiProtocol)
@requires(CPU)
class PyInputAdapter(PyLoihiProcessModel):
    """
    Python implementation of InputAdapter for CPU execution.
    Simply passes data through from input to output port.
    """
    inp: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)

    def run_spk(self):
        data = self.inp.recv()
        self.out.send(data)


@implements(proc=OutputAdapter, protocol=LoihiProtocol)
@requires(CPU)
class PyOutputAdapter(PyLoihiProcessModel):
    """
    Python implementation of OutputAdapter for CPU execution.
    Simply passes data through from input to output port.
    """
    inp: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)

    def run_spk(self):
        data = self.inp.recv()
        self.out.send(data)


# Loihi2 Hardware ProcessModels for Adapters
if LOIHI2_AVAILABLE:
    @implements(proc=InputAdapter, protocol=LoihiProtocol)
    @requires(Loihi2NeuroCore)
    class NxInputAdapter(AbstractSubProcessModel):
        """
        Loihi2 implementation of InputAdapter using embedded_io.
        Transfers spikes from CPU Python process to Loihi2 neuro-cores.
        """
        def __init__(self, proc: AbstractProcess):
            shape = proc.proc_params.get("shape")
            self.adapter = eio.spike.PyToNxAdapter(shape=shape)
            proc.inp.connect(self.adapter.inp)
            self.adapter.out.connect(proc.out)

    @implements(proc=OutputAdapter, protocol=LoihiProtocol)
    @requires(Loihi2NeuroCore)
    class NxOutputAdapter(AbstractSubProcessModel):
        """
        Loihi2 implementation of OutputAdapter using embedded_io.
        Transfers spikes from Loihi2 neuro-cores to CPU Python process.
        """
        def __init__(self, proc: AbstractProcess):
            shape = proc.proc_params.get("shape")
            self.adapter = eio.spike.NxToPyAdapter(shape=shape)
            proc.inp.connect(self.adapter.inp)
            self.adapter.out.connect(proc.out)


# ============================================================================
# Input Encoding: Spectrogram to Spike Process
# ============================================================================

class SpectrogramToSpike(AbstractProcess):
    """
    Input process to convert spectrograms to binary spikes via rate encoding.

    The spectrogram pixels are normalized between [-1, 1] (from dataset transforms).
    They are rate-encoded to spikes using: J = gain * pixel + bias
    where J is the input current to the encoding neuron.
    """
    def __init__(self, img_shape, n_tsteps, curr_img_id, dataset, v_thr=1, gain=1, bias=0):
        # Handle both scalar (flattened) and tuple (multi-dimensional) img_shape
        if isinstance(img_shape, tuple):
            output_shape = img_shape
            flat_size = np.prod(img_shape)
        else:
            output_shape = (img_shape,)
            flat_size = img_shape

        # Pass dataset and output_shape as proc_params (not as Vars)
        super().__init__(dataset=dataset, output_shape=output_shape)

        self.spk_out = OutPort(shape=output_shape)
        self.label_out = OutPort(shape=(flat_size,))

        self.curr_img_id = Var(shape=(1,), init=curr_img_id)
        self.n_ts = Var(shape=(1,), init=n_tsteps)
        self.inp_img = Var(shape=(flat_size,))  # Always flattened internally
        self.ground_truth_mask = Var(shape=(flat_size,))  # Flattened ground truth mask
        self.v = Var(shape=(flat_size,), init=0)
        self.vth = Var(shape=(1,), init=v_thr)
        self.gain = Var(shape=(1,), init=gain)
        self.bias = Var(shape=(1,), init=bias)
        # Dataset and output_shape are NOT stored as Vars - they're passed through proc_params


@implements(proc=SpectrogramToSpike, protocol=LoihiProtocol)
@requires(CPU)
class PySpectrogramToSpikeModel(PyLoihiProcessModel):
    """
    Python implementation of SpectrogramToSpike for CPU execution.
    Performs rate encoding of normalized spectrogram pixels to spikes.
    """
    spk_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, bool, precision=1)
    label_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int32, precision=32)

    curr_img_id: int = LavaPyType(int, int, precision=32)
    n_ts: int = LavaPyType(int, int, precision=32)
    inp_img: np.ndarray = LavaPyType(np.ndarray, float)
    ground_truth_mask: np.ndarray = LavaPyType(np.ndarray, float)
    v: np.ndarray = LavaPyType(np.ndarray, float)
    vth: float = LavaPyType(float, float)
    gain: float = LavaPyType(float, float)
    bias: float = LavaPyType(float, float)
    # Dataset and output_shape are NOT Vars - they're stored directly in the ProcessModel

    def __init__(self, proc_params):
        super().__init__(proc_params=proc_params)
        # Store dataset and output_shape as regular Python attributes (not Vars)
        self.dataset = proc_params['dataset']
        self.output_shape = proc_params['output_shape']

    def post_guard(self):
        """
        Guard function for post-management phase.
        Returns True after n_ts steps to load next image.
        """
        if self.time_step % self.n_ts == 1:
            return True
        return False

    def run_post_mgmt(self):
        """
        Post-management phase: Load next image and ground truth.
        Executed when post_guard() returns True.
        """
        if self.curr_img_id < len(self.dataset):
            # Load spectrogram and mask from dataset
            spectrogram, mask = self.dataset[self.curr_img_id]

            # Convert to numpy and flatten
            # Spectrogram: (1, H, W) -> (H*W,)
            self.inp_img = spectrogram.numpy().flatten().astype(float)

            # Mask: (1, H, W) -> (H*W,) with values 0 or 1
            self.ground_truth_mask = mask.numpy().flatten().astype(float)

            # Send ground truth mask for evaluation (flattened)
            self.label_out.send(self.ground_truth_mask.astype(np.int32))

            # Reset neuron voltage
            self.v = np.zeros(self.v.shape, dtype=float)
            self.curr_img_id += 1
        else:
            # No more images, send zeros
            self.inp_img = np.zeros(self.inp_img.shape, dtype=float)
            self.v = np.zeros(self.v.shape, dtype=float)

    def run_spk(self):
        """
        Spiking phase: Rate encode pixels to spikes.
        Executed every time-step.
        """
        # Rate encoding: J = gain * pixel + bias
        J = self.gain * self.inp_img + self.bias
        self.v[:] = self.v[:] + J[:]

        # Spike when voltage exceeds threshold
        mask = self.v > self.vth
        self.v[mask] = 0  # Reset spiked neurons

        # Reshape to match output port shape (handles both 1D and 3D shapes)
        if len(self.output_shape) > 1:
            # Multi-dimensional output expected, reshape flattened mask
            mask_reshaped = mask.reshape(tuple(self.output_shape))
            self.spk_out.send(mask_reshaped)
        else:
            # 1D output, send as-is
            self.spk_out.send(mask)


# ============================================================================
# Reshape Process for flattening multi-dimensional outputs
# ============================================================================

class Flatten(AbstractProcess):
    """
    Process to flatten multi-dimensional arrays.
    """
    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.inp = InPort(shape=input_shape)
        self.out = OutPort(shape=output_shape)


@implements(proc=Flatten, protocol=LoihiProtocol)
@requires(CPU)
class PyFlatten(PyLoihiProcessModel):
    """
    Python implementation of Flatten for CPU execution.
    """
    inp: PyInPort = LavaPyType(PyInPort.VEC_DENSE, bool, precision=1)
    out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, bool, precision=1)

    def run_spk(self):
        data = self.inp.recv()
        self.out.send(data.flatten())


# ============================================================================
# Output Decoding: Spike to Segmentation Mask Process
# ============================================================================

class SpikeToMask(AbstractProcess):
    """
    Output process to collect output neuron spikes and convert to segmentation masks.

    The network outputs 2 channels (background, RFI).
    Spikes are accumulated over n_tsteps, then argmax is applied to get binary mask.
    """
    def __init__(self, n_tsteps, num_test_imgs, output_shape, spatial_shape):
        """
        Args:
            n_tsteps: Number of timesteps per image
            num_test_imgs: Total number of test images
            output_shape: Shape of network output (2-channel flattened, e.g., 2*64*64)
            spatial_shape: Spatial dimensions (H, W) for reshaping, e.g., (64, 64)
        """
        super().__init__()
        self.spikes_in = InPort(shape=output_shape)  # Receives output spikes (2-channel)
        self.label_in = InPort(shape=(spatial_shape[0] * spatial_shape[1],))  # Receives ground truth (flattened)

        self.spikes_accum = Var(shape=output_shape)  # Accumulate spikes for prediction
        self.n_ts = Var(shape=(1,), init=n_tsteps)
        self.spatial_shape = Var(shape=(2,), init=spatial_shape)  # (H, W)
        self.pred_masks = Var(shape=(num_test_imgs, spatial_shape[0] * spatial_shape[1]))
        self.true_masks = Var(shape=(num_test_imgs, spatial_shape[0] * spatial_shape[1]))


@implements(proc=SpikeToMask, protocol=LoihiProtocol)
@requires(CPU)
class PySpikeToMaskModel(PyLoihiProcessModel):
    """
    Python implementation of SpikeToMask for CPU execution.
    Accumulates spikes and applies argmax to get binary segmentation mask.
    """
    spikes_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, bool, precision=1)
    label_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, int, precision=32)
    spikes_accum: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=32)
    n_ts: int = LavaPyType(int, int, precision=32)
    spatial_shape: np.ndarray = LavaPyType(np.ndarray, int, precision=32)
    pred_masks: np.ndarray = LavaPyType(np.ndarray, int, precision=32)
    true_masks: np.ndarray = LavaPyType(np.ndarray, int, precision=32)

    def __init__(self, proc_params):
        super().__init__(proc_params=proc_params)
        self.curr_idx = 0

    def post_guard(self):
        """
        Guard function for Post-Management phase.
        Returns True after every n_ts timesteps (when an image is fully processed).
        """
        if self.time_step % self.n_ts == 0 and self.time_step > 0:
            return True
        return False

    def run_post_mgmt(self):
        """
        Post-Management phase: Compute prediction from accumulated spikes.
        Executed when post_guard() returns True.
        """
        # Receive ground truth mask
        true_mask = self.label_in.recv()  # Shape: (H*W,)
        self.true_masks[self.curr_idx] = true_mask

        # Reshape accumulated spikes: (2*H*W,) -> (2, H*W)
        H, W = int(self.spatial_shape[0]), int(self.spatial_shape[1])
        spikes_2d = self.spikes_accum.reshape(2, H * W)

        # Apply argmax across channels to get binary mask
        pred_mask = np.argmax(spikes_2d, axis=0)  # Shape: (H*W,)
        self.pred_masks[self.curr_idx] = pred_mask

        # Move to next image
        self.curr_idx += 1

        # Reset spike accumulator
        self.spikes_accum = np.zeros_like(self.spikes_accum)

    def run_spk(self):
        """
        Spiking phase: Accumulate incoming spikes.
        Executed every time-step.
        """
        spk_in = self.spikes_in.recv()
        self.spikes_accum = self.spikes_accum + spk_in.astype(np.int32)
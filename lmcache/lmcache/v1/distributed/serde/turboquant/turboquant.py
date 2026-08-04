# SPDX-License-Identifier: Apache-2.0
"""
TurboQuant serde backend for LMCache.

This backend is intended to compress LMCache KV tensors before L2 store and
decompress them after L2 load.

Supported presets:
- turboquant_k8v4
- turboquant_4bit_nc
- turboquant_k3v4_nc
- turboquant_3bit_nc

Input KV layout: [2, num_layers, num_tokens, hidden_dim]
Serialized layout: [num_layers, num_blocks, block_size, num_heads, slot_size]
"""

# Standard
from dataclasses import dataclass
from functools import lru_cache
import math
import subprocess

# Third Party
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.serde.async_processor import AsyncSerdeProcessor
from lmcache.v1.distributed.serde.base import Deserializer, SerdeProcessor, Serializer
from lmcache.v1.distributed.serde.factory import register_serde_factory
from lmcache.v1.memory_management import MemoryObj

TQ_PRESETS: dict[str, dict[str, object]] = {
    "turboquant_k8v4": {
        "key_quant_bits": 8,
        "value_quant_bits": 4,
        "norm_correction": False,
    },
    "turboquant_4bit_nc": {
        "key_quant_bits": 4,
        "value_quant_bits": 4,
        "norm_correction": True,
    },
    "turboquant_k3v4_nc": {
        "key_quant_bits": 3,
        "value_quant_bits": 4,
        "norm_correction": True,
    },
    "turboquant_3bit_nc": {
        "key_quant_bits": 3,
        "value_quant_bits": 3,
        "norm_correction": True,
    },
}

_TQ_LAYER_SEED_STRIDE = 1337


@dataclass(frozen=True)
class TurboQuantSerdeConfig:
    """Configuration for TurboQuant serde.

    Args:
        preset: TurboQuant compression preset.
        head_dim: Per-head hidden dimension.
        block_size: Token block size used by the compressed layout.
        skip_first_layers: Number of leading layers stored without quantization.
        skip_last_layers: Number of trailing layers stored without quantization.
        cuda_device: CUDA staging device used when both source and destination
            tensors are CPU tensors. Empty string means automatically select a
            CUDA device with sufficient free memory and the lowest GPU
            utilization.
    """

    preset: str = "turboquant_k8v4"
    head_dim: int = 128
    block_size: int = 16
    skip_first_layers: int = 2
    skip_last_layers: int = 2
    cuda_device: str = ""

    def __post_init__(self) -> None:
        if self.skip_first_layers < 0 or self.skip_last_layers < 0:
            raise ValueError("TurboQuant skipped layer counts must be non-negative")

    @property
    def _preset_config(self) -> dict[str, object]:
        if self.preset not in TQ_PRESETS:
            valid = ", ".join(TQ_PRESETS)
            raise ValueError(
                f"Unsupported TurboQuant preset: {self.preset!r}. "
                f"Valid presets: {valid}"
            )
        return TQ_PRESETS[self.preset]

    @property
    def key_quant_bits(self) -> int:
        return int(self._preset_config["key_quant_bits"])  # type: ignore[call-overload]

    @property
    def key_fp8(self) -> bool:
        return self.key_quant_bits == 8

    @property
    def key_mse_bits(self) -> int:
        if self.key_fp8:
            return 0
        return self.key_quant_bits

    @property
    def value_quant_bits(self) -> int:
        return int(self._preset_config["value_quant_bits"])  # type: ignore[call-overload]

    @property
    def effective_value_quant_bits(self) -> int:
        return self.value_quant_bits

    @property
    def norm_correction(self) -> bool:
        return bool(self._preset_config["norm_correction"])

    @property
    def mse_bits(self) -> int:
        if self.key_fp8:
            return self.value_quant_bits
        return self.key_quant_bits

    @property
    def centroid_bits(self) -> int:
        return self.mse_bits

    @property
    def n_centroids(self) -> int:
        return 2**self.mse_bits

    @property
    def key_packed_size(self) -> int:
        if self.key_fp8:
            return self.head_dim
        mse_bytes = math.ceil(self.head_dim * self.key_mse_bits / 8)
        norm_bytes = 2
        return mse_bytes + norm_bytes

    @property
    def value_packed_size(self) -> int:
        data_bytes = math.ceil(self.head_dim * self.value_quant_bits / 8)
        return data_bytes + 4  # scale fp16 + zero fp16

    @property
    def slot_size(self) -> int:
        return self.key_packed_size + self.value_packed_size

    @property
    def slot_size_aligned(self) -> int:
        s = self.slot_size
        return s + (s % 2)


def _validate_layout_shape(
    shape: torch.Size, cfg: TurboQuantSerdeConfig
) -> tuple[int, int, int, int]:
    """Validate LMCache KV layout and return L, T, H, D.

    Expected input shape:
        [2, num_layers, num_tokens, hidden_dim]

    Returns:
        num_layers, num_tokens, num_heads, head_dim
    """
    if len(shape) != 4:
        raise ValueError(
            "TurboQuant serde expects 4D KV tensor "
            f"[2, L, T, hidden_dim], got {tuple(shape)}"
        )
    if int(shape[0]) != 2:
        raise ValueError(
            f"TurboQuant serde expects first dim kv_size=2, got {int(shape[0])}"
        )

    num_layers = int(shape[1])
    num_tokens = int(shape[2])
    hidden_dim = int(shape[3])
    head_dim = cfg.head_dim

    if hidden_dim % head_dim != 0:
        raise ValueError(
            f"hidden_dim={hidden_dim} must be divisible by head_dim={head_dim}"
        )

    num_heads = hidden_dim // head_dim
    return num_layers, num_tokens, num_heads, head_dim


def _layer_ranges(num_layers: int, cfg: TurboQuantSerdeConfig) -> tuple[int, int]:
    """Return the half-open range of layers compressed with TurboQuant."""
    quant_start = min(cfg.skip_first_layers, num_layers)
    quant_end = max(quant_start, num_layers - cfg.skip_last_layers)
    return quant_start, quant_end


def _raw_group_nbytes(
    shape: torch.Size,
    dtype: torch.dtype,
    num_layers: int,
) -> int:
    _, _, num_tokens, hidden_dim = shape
    return (
        2
        * num_layers
        * int(num_tokens)
        * int(hidden_dim)
        * torch.empty((), dtype=dtype).element_size()
    )


def _serialized_nbytes_for_shape(
    shape: torch.Size,
    dtype: torch.dtype,
    cfg: TurboQuantSerdeConfig,
) -> int:
    """Return serialized size in bytes for one LMCache KV tensor."""
    num_layers, num_tokens, num_heads, _ = _validate_layout_shape(shape, cfg)
    num_blocks = math.ceil(num_tokens / cfg.block_size)
    quant_start, quant_end = _layer_ranges(num_layers, cfg)
    quant_layers = quant_end - quant_start
    raw_layers = num_layers - quant_layers
    return _raw_group_nbytes(shape, dtype, raw_layers) + (
        quant_layers * num_blocks * cfg.block_size * num_heads * cfg.slot_size_aligned
    )


def _compressed_layout_for_shape(
    shape: torch.Size,
    cfg: TurboQuantSerdeConfig,
    num_layers: int | None = None,
) -> tuple[int, int, int, int, int]:
    """Return compressed layout [L, num_blocks, block_size, H, slot_size]."""
    total_layers, num_tokens, num_heads, _ = _validate_layout_shape(shape, cfg)
    if num_layers is None:
        num_layers = total_layers
    num_blocks = math.ceil(num_tokens / cfg.block_size)
    return (
        num_layers,
        num_blocks,
        cfg.block_size,
        num_heads,
        cfg.slot_size_aligned,
    )


def _make_slot_mapping(num_tokens: int, device: torch.device) -> torch.Tensor:
    """Sequential slot mapping: token i -> slot i."""
    return torch.arange(num_tokens, device=device, dtype=torch.int32)


def _make_block_table(num_blocks: int, device: torch.device) -> torch.Tensor:
    """Sequential block table: logical block i -> physical block i."""
    return torch.arange(num_blocks, device=device, dtype=torch.int32).view(
        1, num_blocks
    )


def _generate_wht_signs(
    head_dim: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """Generate deterministic random ±1 signs for WHT rotation."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    bits = torch.randint(0, 2, (head_dim,), generator=gen, device="cpu")
    signs = bits.float() * 2 - 1
    return signs.to(device=device, dtype=torch.float32)


@lru_cache(maxsize=32)
def _solve_lloyd_max(
    head_dim: int,
    bits: int,
    max_iter: int = 200,
    tol: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve Lloyd-Max optimal quantizer for N(0, 1/head_dim)."""
    n_levels = 2**bits
    sigma2 = 1.0 / head_dim
    sigma = math.sqrt(sigma2)

    def gaussian_pdf(x: float) -> float:
        return (1.0 / math.sqrt(2 * math.pi * sigma2)) * math.exp(-x * x / (2 * sigma2))

    def trapz(f, a: float, b: float, n: int = 200) -> float:
        h = (b - a) / n
        result = 0.5 * (f(a) + f(b))
        for i in range(1, n):
            result += f(a + i * h)
        return result * h

    lo, hi = -3.5 * sigma, 3.5 * sigma
    centroids = [lo + (hi - lo) * (i + 0.5) / n_levels for i in range(n_levels)]

    for _ in range(max_iter):
        boundaries = [
            (centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)
        ]
        edges = [lo * 3] + boundaries + [hi * 3]

        new_centroids = []
        for i in range(n_levels):
            a, b = edges[i], edges[i + 1]
            num = trapz(lambda x: x * gaussian_pdf(x), a, b)
            den = trapz(gaussian_pdf, a, b)
            new_centroids.append(num / den if den > 1e-15 else centroids[i])

        if max(abs(new_centroids[i] - centroids[i]) for i in range(n_levels)) < tol:
            break
        centroids = new_centroids

    boundaries = [(centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)]

    return (
        torch.tensor(centroids, dtype=torch.float32),
        torch.tensor(boundaries, dtype=torch.float32),
    )


@lru_cache(maxsize=32)
def _build_hadamard_cpu(head_dim: int) -> torch.Tensor:
    """Build an orthonormal Sylvester Hadamard matrix on CPU."""
    if head_dim <= 0 or head_dim & (head_dim - 1) != 0:
        raise ValueError(
            "TurboQuant WHT rotation requires head_dim to be a power of two, "
            f"got {head_dim}"
        )

    h = torch.tensor([[1.0]], dtype=torch.float32)
    while h.shape[0] < head_dim:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / math.sqrt(head_dim)


def _make_tq_tensors_for_layer(
    cfg: TurboQuantSerdeConfig,
    layer_idx: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create PiT, midpoints, and centroids for one TurboQuant layer."""
    seed = 42 + layer_idx * _TQ_LAYER_SEED_STRIDE
    signs = _generate_wht_signs(cfg.head_dim, seed=seed, device=device)
    hadamard = _build_hadamard_cpu(cfg.head_dim).to(device=device)
    pi_t = (signs.unsqueeze(1) * hadamard).contiguous()

    centroids_cpu, midpoints_cpu = _solve_lloyd_max(
        cfg.head_dim,
        cfg.centroid_bits,
    )
    centroids = centroids_cpu.to(device=device, dtype=torch.float32)
    midpoints = midpoints_cpu.to(device=device, dtype=torch.float32)

    if cfg.key_fp8:
        # FP8-key store/dequant paths do not use PiT/midpoints/centroids, but
        # keep valid tensors to satisfy kernel signatures.
        midpoints = torch.empty((0,), device=device, dtype=torch.float32)
        centroids = torch.empty((1,), device=device, dtype=torch.float32)

    return pi_t, midpoints, centroids


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    """Return the number of bytes occupied by a tensor.

    Args:
        tensor: Tensor whose storage size should be estimated.

    Returns:
        Number of bytes represented by the tensor shape and element size.
    """
    return tensor.numel() * tensor.element_size()


def _cuda_utilization(device_index: int) -> int:
    """Return GPU utilization percentage for a CUDA device.

    Args:
        device_index: CUDA device index.

    Returns:
        GPU utilization percentage. Returns 0 if utilization cannot be queried.
    """
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={device_index}",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return int(out.strip().splitlines()[0])
    except Exception:
        return 0


def _normalize_cuda_device(cuda_device: str) -> torch.device:
    """Normalize a configured CUDA device string.

    Args:
        cuda_device: CUDA device string, such as "0", "cuda", or "cuda:0".

    Returns:
        Normalized torch CUDA device.

    Raises:
        ValueError: If the configured device is not a CUDA device.
    """
    if cuda_device.isdigit():
        return torch.device(f"{torch_device_type}:{cuda_device}")

    device = torch.device(cuda_device)
    if device.type != torch_device_type:
        raise ValueError(
            f"TurboQuant cuda_device must be a CUDA device, got {cuda_device!r}"
        )
    if device.index is None:
        return torch.device(torch_device_type, torch_dev.current_device())
    return torch.device(torch_device_type, device.index)


def _auto_select_cuda_device(required_bytes: int) -> torch.device:
    """Select a CUDA device with enough free memory and lowest utilization.

    Args:
        required_bytes: Estimated temporary CUDA memory required by the serde
            operation.

    Returns:
        Selected CUDA device.

    Raises:
        RuntimeError: If CUDA is unavailable or no device has enough free memory.
    """
    if not torch_dev.is_available():
        raise RuntimeError("TurboQuant Triton serde requires CUDA")

    candidates: list[tuple[int, int, int]] = []
    for device_index in range(torch_dev.device_count()):
        with torch_dev.device(device_index):
            free_bytes, _ = torch_dev.mem_get_info()

        if int(free_bytes) < required_bytes:
            continue

        util = _cuda_utilization(device_index)
        candidates.append((util, -int(free_bytes), device_index))

    if not candidates:
        raise RuntimeError(
            "No CUDA device has enough free memory for TurboQuant serde staging: "
            f"required_bytes={required_bytes}"
        )

    _, _, selected = min(candidates)
    torch_dev.set_device(selected)
    return torch.device(torch_device_type, selected)


def _select_cuda_device(
    required_bytes: int,
    configured_cuda_device: str,
    *tensors: torch.Tensor,
) -> torch.device:
    """Select the CUDA device for TurboQuant Triton kernels.

    Args:
        required_bytes: Estimated temporary CUDA memory required by the serde
            operation.
        configured_cuda_device: Explicit CUDA device from config. Empty string
            means auto-select when CPU staging is needed.
        tensors: Source and destination tensors participating in the serde
            operation.

    Returns:
        CUDA device used for Triton kernels and temporary staging.

    Raises:
        RuntimeError: If CUDA is unavailable or no device has enough memory.
        ValueError: If CUDA tensors are on different devices, or if the
            configured CUDA device conflicts with existing CUDA tensors.
    """
    cuda_devices = {tensor.device for tensor in tensors if tensor.is_cuda}

    if len(cuda_devices) > 1:
        devices = ", ".join(sorted(str(device) for device in cuda_devices))
        raise ValueError(
            "TurboQuant serde requires all CUDA tensors in one operation "
            f"to be on the same device, got: {devices}"
        )

    if configured_cuda_device:
        configured = _normalize_cuda_device(configured_cuda_device)
        if cuda_devices and configured not in cuda_devices:
            existing = next(iter(cuda_devices))
            raise ValueError(
                "Configured TurboQuant cuda_device conflicts with tensor device: "
                f"configured={configured}, tensor_device={existing}"
            )
        return configured

    if cuda_devices:
        selected = next(iter(cuda_devices))
        if selected.index is None:
            selected = torch.device(torch_device_type, torch_dev.current_device())
        else:
            selected = torch.device(torch_device_type, selected.index)
        torch_dev.set_device(selected)
        return selected

    return _auto_select_cuda_device(required_bytes)


class TurboQuantSerializer(Serializer):
    """TurboQuant serializer skeleton."""

    def __init__(self, cfg: TurboQuantSerdeConfig):
        self._cfg = cfg

    def serialize(self, src: MemoryObj, dst: MemoryObj, key: ObjectKey) -> int:
        """Serialize a KV tensor into a TurboQuant-compressed byte buffer.

        Args:
            src: Source memory object containing a KV tensor with shape
                ``[2, num_layers, num_tokens, hidden_dim]``.
            dst: Destination memory object containing a ``torch.uint8`` tensor
                used as the serialized byte buffer.
            key: Object key for this pair (unused: TurboQuant is
                content-agnostic).

        Returns:
            The number of serialized bytes written to ``dst``.

        Raises:
            ValueError: If source or destination tensors are missing, if the
                destination buffer is too small, if the destination dtype is not
                ``torch.uint8``, or if the KV tensor layout is unsupported.
            RuntimeError: If CUDA is unavailable or no CUDA device has enough
                memory for TurboQuant staging.
        """
        src_tensor = src.tensor
        dst_tensor = dst.tensor
        if src_tensor is None or dst_tensor is None:
            raise ValueError("TurboQuant serde requires src and dst to have tensors")

        n_bytes = _serialized_nbytes_for_shape(
            src_tensor.shape, src_tensor.dtype, self._cfg
        )
        if dst_tensor.numel() < n_bytes:
            raise ValueError(
                f"Destination buffer too small: got {dst_tensor.numel()} bytes, "
                f"need {n_bytes}"
            )

        if dst_tensor.dtype != torch.uint8:
            raise ValueError(
                "TurboQuant serialized destination must be torch.uint8, "
                f"got {dst_tensor.dtype}"
            )
        required_cuda_bytes = n_bytes
        if not src_tensor.is_cuda:
            required_cuda_bytes += _tensor_nbytes(src_tensor)
        if not dst_tensor.is_cuda:
            required_cuda_bytes += n_bytes

        cuda_device = _select_cuda_device(
            required_cuda_bytes,
            self._cfg.cuda_device,
            src_tensor,
            dst_tensor,
        )

        # StorageManager may provide CPU / pinned-memory MemoryObjs. Triton
        # kernels require CUDA tensors, so use temporary CUDA buffers when
        # necessary and copy the serialized bytes back to the original dst.
        src_work = (
            src_tensor
            if src_tensor.is_cuda
            else src_tensor.clone().to(device=cuda_device)
        )
        dst_work = (
            dst_tensor
            if dst_tensor.is_cuda
            else torch.empty(n_bytes, dtype=torch.uint8, device=cuda_device)
        )

        cfg = self._cfg
        num_layers, num_tokens, num_heads, head_dim = _validate_layout_shape(
            src_work.shape, cfg
        )
        quant_start, quant_end = _layer_ranges(num_layers, cfg)
        quant_layers = quant_end - quant_start
        dst_flat = dst_work.flatten()[:n_bytes]
        offset = 0

        first_raw_bytes = _raw_group_nbytes(src_work.shape, src_work.dtype, quant_start)
        if first_raw_bytes:
            raw = src_work[:, :quant_start].contiguous().view(torch.uint8).flatten()
            dst_flat[:first_raw_bytes].copy_(raw)
            offset = first_raw_bytes

        if quant_layers:
            # First Party
            from lmcache.v1.distributed.serde.turboquant.store_kernel import (
                triton_turboquant_store,
            )

            compressed_shape = _compressed_layout_for_shape(
                src_work.shape, cfg, quant_layers
            )
            compressed_bytes = math.prod(compressed_shape)
            dst_view = dst_flat[offset : offset + compressed_bytes].view(
                *compressed_shape
            )
            slot_mapping = _make_slot_mapping(num_tokens, cuda_device)

            # LMCache layout: [2, L, T, hidden_dim]
            # Kernel input layout per layer: key/value [T, H, D]
            for compressed_idx, layer_idx in enumerate(range(quant_start, quant_end)):
                k_tensor = (
                    src_work[0, layer_idx]
                    .view(num_tokens, num_heads, head_dim)
                    .contiguous()
                )
                value = (
                    src_work[1, layer_idx]
                    .view(num_tokens, num_heads, head_dim)
                    .contiguous()
                )
                pi_t, midpoints, _ = _make_tq_tensors_for_layer(
                    cfg, layer_idx, cuda_device
                )
                triton_turboquant_store(
                    k_tensor,
                    value,
                    dst_view[compressed_idx],
                    slot_mapping,
                    pi_t,
                    midpoints,
                    mse_bits=cfg.key_mse_bits,
                    key_packed_size=cfg.key_packed_size,
                    value_quant_bits=cfg.value_quant_bits,
                    key_fp8=cfg.key_fp8,
                )
            offset += compressed_bytes

        if quant_end < num_layers:
            raw = src_work[:, quant_end:].contiguous().view(torch.uint8).flatten()
            dst_flat[offset : offset + raw.numel()].copy_(raw)

        if not dst_tensor.is_cuda:
            dst_tensor.flatten()[:n_bytes].copy_(dst_work.cpu().flatten()[:n_bytes])

        return n_bytes

    def estimate_serialized_size(self, layout_desc: MemoryLayoutDesc) -> int:
        total = 0
        for shape, dtype in zip(layout_desc.shapes, layout_desc.dtypes, strict=False):
            total += _serialized_nbytes_for_shape(shape, dtype, self._cfg)
        return total


class TurboQuantDeserializer(Deserializer):
    """TurboQuant deserializer skeleton."""

    def __init__(self, cfg: TurboQuantSerdeConfig):
        self._cfg = cfg

    def deserialize(self, src: MemoryObj, dst: MemoryObj, key: ObjectKey) -> None:
        """Deserialize a TurboQuant byte buffer into a destination KV tensor.

        Args:
            src: Source memory object containing the serialized TurboQuant
                byte buffer as a ``torch.uint8`` tensor.
            dst: Destination memory object containing the reconstructed KV
                tensor with shape ``[2, num_layers, num_tokens, hidden_dim]``.
            key: Object key for this pair (unused: TurboQuant is
                content-agnostic).

        Raises:
            ValueError: If source or destination tensors are missing, if the
                source buffer is too small, if the source dtype is not
                ``torch.uint8``, or if the destination KV tensor layout is
                unsupported.
            RuntimeError: If CUDA is unavailable or no CUDA device has enough
                memory for TurboQuant staging.
        """
        src_tensor = src.tensor
        dst_tensor = dst.tensor
        if src_tensor is None or dst_tensor is None:
            raise ValueError("TurboQuant serde requires src and dst to have tensors")

        n_bytes = _serialized_nbytes_for_shape(
            dst_tensor.shape, dst_tensor.dtype, self._cfg
        )
        if src_tensor.numel() < n_bytes:
            raise ValueError(
                f"Source buffer too small: got {src_tensor.numel()} bytes, "
                f"need {n_bytes}"
            )

        if src_tensor.dtype != torch.uint8:
            raise ValueError(
                "TurboQuant serialized source must be torch.uint8, "
                f"got {src_tensor.dtype}"
            )
        required_cuda_bytes = n_bytes
        if not src_tensor.is_cuda:
            required_cuda_bytes += n_bytes
        if not dst_tensor.is_cuda:
            required_cuda_bytes += _tensor_nbytes(dst_tensor)

        cuda_device = _select_cuda_device(
            required_cuda_bytes,
            self._cfg.cuda_device,
            src_tensor,
            dst_tensor,
        )

        # StorageManager may provide CPU / pinned-memory MemoryObjs. Triton
        # kernels require CUDA tensors, so copy compressed bytes to CUDA and
        # dequantize into a CUDA temporary when the destination is CPU.
        src_work = (
            src_tensor
            if src_tensor.is_cuda
            else src_tensor.flatten()[:n_bytes].clone().to(device=cuda_device)
        )
        dst_work = (
            dst_tensor
            if dst_tensor.is_cuda
            else torch.empty(
                dst_tensor.shape,
                dtype=dst_tensor.dtype,
                device=cuda_device,
            )
        )

        cfg = self._cfg
        num_layers, num_tokens, num_heads, head_dim = _validate_layout_shape(
            dst_work.shape, cfg
        )
        hidden_dim = num_heads * head_dim
        quant_start, quant_end = _layer_ranges(num_layers, cfg)
        quant_layers = quant_end - quant_start
        src_flat = src_work.flatten()[:n_bytes]
        offset = 0

        first_raw_bytes = _raw_group_nbytes(dst_work.shape, dst_work.dtype, quant_start)
        if first_raw_bytes:
            raw = torch.empty(
                (2, quant_start, num_tokens, hidden_dim),
                dtype=dst_work.dtype,
                device=cuda_device,
            )
            raw.view(torch.uint8).flatten().copy_(src_flat[:first_raw_bytes])
            dst_work[:, :quant_start].copy_(raw)
            offset = first_raw_bytes

        if quant_layers:
            # First Party
            from lmcache.v1.distributed.serde.turboquant.decode_kernel import (
                _tq_full_dequant_kv,
                _use_fp8_e4b15,
            )

            compressed_shape = _compressed_layout_for_shape(
                dst_work.shape, cfg, quant_layers
            )
            compressed_bytes = math.prod(compressed_shape)
            src_view = src_flat[offset : offset + compressed_bytes].view(
                *compressed_shape
            )
            num_blocks = compressed_shape[1]
            alloc_len = num_blocks * cfg.block_size
            block_table = _make_block_table(num_blocks, cuda_device)
            block_d = 1 << (head_dim - 1).bit_length()
            val_data_bytes = math.ceil(head_dim * cfg.value_quant_bits / 8)
            mse_bytes = (
                math.ceil(head_dim * cfg.key_mse_bits / 8)
                if not cfg.key_fp8
                else head_dim
            )
            k_out = torch.empty(
                (1, num_heads, alloc_len, head_dim),
                dtype=torch.float16,
                device=cuda_device,
            )
            v_out = torch.empty_like(k_out)

            for compressed_idx, layer_idx in enumerate(range(quant_start, quant_end)):
                kv_cache_layer = src_view[compressed_idx]
                pi_t, _, centroids = _make_tq_tensors_for_layer(
                    cfg, layer_idx, cuda_device
                )
                grid = (alloc_len, num_heads)
                _tq_full_dequant_kv[grid](
                    kv_cache_layer,
                    block_table,
                    centroids,
                    k_out,
                    v_out,
                    k_out.stride(0),
                    k_out.stride(1),
                    k_out.stride(2),
                    v_out.stride(0),
                    v_out.stride(1),
                    v_out.stride(2),
                    kv_cache_layer.stride(0),
                    kv_cache_layer.stride(1),
                    kv_cache_layer.stride(2),
                    block_table.stride(0),
                    HEAD_DIM=head_dim,
                    BLOCK_SIZE=cfg.block_size,
                    NUM_KV_HEADS=num_heads,
                    MSE_BYTES=mse_bytes,
                    KPS=cfg.key_packed_size,
                    VQB=cfg.value_quant_bits,
                    VAL_DATA_BYTES=val_data_bytes,
                    MSE_BITS=cfg.key_mse_bits,
                    KEY_FP8=1 if cfg.key_fp8 else 0,
                    BLOCK_D=block_d,
                    NORM_CORRECTION=1 if cfg.norm_correction else 0,
                    FP8_E4B15=_use_fp8_e4b15(cuda_device.index or 0),
                    num_warps=4,
                )

                k_layer = k_out[0, :, :num_tokens, :].transpose(0, 1).contiguous()
                if not cfg.key_fp8:
                    # Restore the original key after TurboQuant's WHT rotation.
                    k_layer = torch.matmul(
                        k_layer.to(torch.float32), pi_t.T.contiguous()
                    ).to(k_out.dtype)
                k_tensor = k_layer.contiguous().view(num_tokens, hidden_dim)
                value = (
                    v_out[0, :, :num_tokens, :]
                    .transpose(0, 1)
                    .contiguous()
                    .view(num_tokens, hidden_dim)
                )
                dst_work[0, layer_idx].copy_(k_tensor.to(dst_work.dtype))
                dst_work[1, layer_idx].copy_(value.to(dst_work.dtype))
            offset += compressed_bytes

        last_raw_layers = num_layers - quant_end
        if last_raw_layers:
            last_raw_bytes = _raw_group_nbytes(
                dst_work.shape, dst_work.dtype, last_raw_layers
            )
            raw = torch.empty(
                (2, last_raw_layers, num_tokens, hidden_dim),
                dtype=dst_work.dtype,
                device=cuda_device,
            )
            raw.view(torch.uint8).flatten().copy_(
                src_flat[offset : offset + last_raw_bytes]
            )
            dst_work[:, quant_end:].copy_(raw)

        if not dst_tensor.is_cuda:
            dst_tensor.copy_(dst_work.cpu())


def _create_turboquant_serde(kwargs: dict[str, object]) -> SerdeProcessor:
    preset = str(kwargs.get("preset", "turboquant_k8v4"))
    head_dim = int(kwargs.get("head_dim", 128))  # type: ignore[call-overload]
    block_size = int(kwargs.get("block_size", 16))  # type: ignore[call-overload]
    skip_first_layers = int(
        kwargs.get("skip_first_layers", 2)  # type: ignore[call-overload]
    )
    skip_last_layers = int(
        kwargs.get("skip_last_layers", 2)  # type: ignore[call-overload]
    )
    max_workers = int(kwargs.get("max_workers", 1))  # type: ignore[call-overload]

    cfg = TurboQuantSerdeConfig(
        preset=preset,
        head_dim=head_dim,
        block_size=block_size,
        skip_first_layers=skip_first_layers,
        skip_last_layers=skip_last_layers,
    )

    return AsyncSerdeProcessor(
        TurboQuantSerializer(cfg),
        TurboQuantDeserializer(cfg),
        max_workers=max_workers,
    )


register_serde_factory("turboquant", _create_turboquant_serde)

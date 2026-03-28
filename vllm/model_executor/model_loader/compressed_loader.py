# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
CompressedModelLoader — loads model weights from losslessly compressed
.cbin checkpoints produced by tools/compress_weights.py.

Two modes (controlled by model_loader_extra_config):
  Mode 1 (default, enable_jit_decompress=false):
    Decompress all weights at load time. Transparent to the rest of vLLM.
    Benefit: smaller on-disk footprint / faster download.

  Mode 2 (enable_jit_decompress=true):
    Keep compressed bytes in pinned CPU RAM. For each forward pass of a
    CPU-offloaded module, transfer compressed bytes to GPU then decompress
    in-place via the GPU decompress op (nvCOMP on CUDA, hipCOMP on ROCm;
    falls back to CPU zlib if neither is available). Benefit: lower CPU
    RAM usage when used together with --cpu-offload-gb.

Usage:
    vllm serve /path/to/compressed-model --load-format compressed

    # JIT streaming for large models:
    vllm serve /path/to/compressed-model \\
        --load-format compressed \\
        --model-loader-extra-config '{"enable_jit_decompress":true}' \\
        --cpu-offload-gb 8
"""

from __future__ import annotations

import dataclasses
import json
import time
import zlib
from collections.abc import Generator
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils.parametrize import type_before_parametrizations
from torch.overrides import TorchFunctionMode
from tqdm import tqdm

from vllm.config import ModelConfig
from vllm.config.load import LoadConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

logger = init_logger(__name__)

# Dtype string → torch.dtype
_DTYPE_MAP: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int32": torch.int32,
    "int8": torch.int8,
    "uint8": torch.uint8,
    "int64": torch.int64,
    "float8_e4m3fn": torch.float8_e4m3fn,
    "float8_e5m2": torch.float8_e5m2,
}


def _load_compression_index(model_path: Path) -> dict:
    """Load and validate compression_index.json from a compressed model dir."""
    index_file = model_path / "compression_index.json"
    if not index_file.exists():
        raise FileNotFoundError(
            f"compression_index.json not found in {model_path}. "
            "This does not appear to be a compressed model directory. "
            "Run tools/compress_weights.py to create one."
        )
    with index_file.open() as f:
        index = json.load(f)
    required = {"algorithm", "tensors"}
    missing = required - set(index.keys())
    if missing:
        raise ValueError(
            f"compression_index.json is missing required keys: {missing}"
        )
    return index


# Chunk size for LZ4 — must match _LZ4_CHUNK_SIZE in compress_weights.py
# and DECOMP_CHUNK_SIZE in csrc/rocm/weight_decompress.cu.
_LZ4_CHUNK_SIZE = 65536


def _compress_lz4_chunked(data: bytes) -> bytes:
    """Compress bytes using chunked LZ4 format (GPU-decompressable).

    Format: [4-byte n_chunks][n_chunks × 4-byte comp_sizes][chunk data...]
    """
    import struct

    try:
        import lz4.block as lz4block  # type: ignore[import]
    except ImportError:
        raise ImportError(
            "lz4 is required for LZ4 compression. Install with: pip install lz4"
        )

    chunks = []
    for offset in range(0, len(data), _LZ4_CHUNK_SIZE):
        chunks.append(
            lz4block.compress(data[offset : offset + _LZ4_CHUNK_SIZE],
                              store_size=False)
        )
    n = len(chunks)
    header = struct.pack(f"<I{n}I", n, *(len(c) for c in chunks))
    return header + b"".join(chunks)


def _decompress_lz4_cpu(data: bytes, original_size: int) -> bytes:
    """Decompress chunked LZ4 bytes on CPU."""
    import struct

    try:
        import lz4.block as lz4block  # type: ignore[import]
    except ImportError:
        raise ImportError(
            "lz4 is required for LZ4-compressed models. "
            "Install with: pip install lz4"
        )

    n_chunks = struct.unpack_from("<I", data, 0)[0]
    comp_sizes = struct.unpack_from(f"<{n_chunks}I", data, 4)
    offset = 4 + n_chunks * 4

    result = bytearray()
    for i, cs in enumerate(comp_sizes):
        remaining = original_size - i * _LZ4_CHUNK_SIZE
        uncomp_size = min(remaining, _LZ4_CHUNK_SIZE)
        chunk = lz4block.decompress(data[offset : offset + cs],
                                    uncompressed_size=uncomp_size)
        result.extend(chunk)
        offset += cs

    return bytes(result)


def _decompress_deflate_cpu(data: bytes, original_size: int) -> bytes:
    """Decompress DEFLATE (zlib) bytes on CPU."""
    raw = zlib.decompress(data)
    if len(raw) != original_size:
        raise ValueError(
            f"Decompressed size mismatch: expected {original_size}, got {len(raw)}"
        )
    return raw


def _decompress_zstd_cpu(data: bytes) -> bytes:
    """Decompress Zstd bytes on CPU."""
    try:
        import zstandard as zstd  # type: ignore[import]
    except ImportError:
        raise ImportError(
            "zstandard is required for Zstd-compressed models. "
            "Install with: pip install zstandard"
        )
    return zstd.ZstdDecompressor().decompress(data)


def _bytes_to_tensor(
    raw: bytes,
    shape: list[int],
    dtype_str: str,
) -> torch.Tensor:
    """Reconstruct a torch.Tensor from raw bytes."""
    import numpy as np  # noqa: PLC0415

    dtype = _DTYPE_MAP.get(dtype_str)
    if dtype is None:
        raise ValueError(f"Unsupported dtype in compression index: {dtype_str!r}")

    # dtypes with no direct numpy equivalent: decode via uint8 view
    if dtype_str in ("bfloat16", "float8_e4m3fn", "float8_e5m2"):
        arr = np.frombuffer(raw, dtype=np.uint8)
        t = torch.from_numpy(arr.copy()).view(dtype)
    else:
        np_map: dict[str, Any] = {
            "float32": "float32",
            "float16": "float16",
            "int32": "int32",
            "int8": "int8",
            "uint8": "uint8",
            "int64": "int64",
        }
        arr = np.frombuffer(raw, dtype=np.dtype(np_map[dtype_str]))
        t = torch.from_numpy(arr.copy())

    return t.reshape(shape)


def _decompress_tensor_cpu(
    compressed: bytes,
    meta: dict,
    algorithm: str,
    zipnn_module: Any | None,
) -> torch.Tensor:
    """Decompress a single tensor on CPU (no GPU involvement)."""
    preshuffled = meta.get("preshuffled", False)

    if preshuffled and zipnn_module is not None:
        try:
            raw = zipnn_module.decompress(compressed)
        except Exception:
            preshuffled = False  # fall through

    if not preshuffled:
        if algorithm == "gdeflate":
            raise ValueError(
                "GDeflate-compressed models require GPU (nvCOMP/hipCOMP) decompression. "
                "Enable JIT mode with gpu_decompress=true, or recompress with "
                "--algorithm lz4."
            )
        if algorithm == "lz4":
            raw = _decompress_lz4_cpu(compressed, meta["original_size"])
        elif algorithm == "deflate":
            raw = _decompress_deflate_cpu(compressed, meta["original_size"])
        elif algorithm == "zstd":
            raw = _decompress_zstd_cpu(compressed)
        else:
            raise ValueError(f"Unsupported compression algorithm: {algorithm!r}")

    return _bytes_to_tensor(raw, meta["shape"], meta["dtype"])


def _try_import_zipnn() -> Any | None:
    try:
        import zipnn  # type: ignore[import]
        return zipnn
    except ImportError:
        return None


def _try_get_gpu_decompress_op():
    """Return the decompress_tensor op if nvCOMP/hipCOMP is available.

    Tries _C (CUDA/nvCOMP) first, then _rocm_C (ROCm/hipCOMP) as fallback.
    Importing the extension registers the ops with PyTorch's dispatch system.
    """
    import torch as _torch

    # Try CUDA path first (_C extension, nvCOMP)
    try:
        import vllm._C  # noqa: F401 — side-effect: registers ops
        op = _torch.ops._C.decompress_tensor
        available = bool(_torch.ops._C.is_gpu_decompress_available())
        return op, available
    except (ImportError, AttributeError, RuntimeError):
        pass

    # Try ROCm path (_rocm_C extension, hipCOMP)
    try:
        import vllm._rocm_C  # noqa: F401 — side-effect: registers ops
        op = _torch.ops._rocm_C.decompress_tensor
        available = bool(_torch.ops._rocm_C.is_gpu_decompress_available())
        return op, available
    except (ImportError, AttributeError, RuntimeError):
        pass

    return None, False


def _try_get_gpu_compress_op():
    """Return the compress_tensor op if nvCOMP/hipCOMP is available.

    Tries _C (CUDA/nvCOMP) first, then _rocm_C (ROCm/hipCOMP) as fallback.
    """
    import torch as _torch

    # Try CUDA path first
    try:
        import vllm._C  # noqa: F401
        op = _torch.ops._C.compress_tensor
        available = bool(_torch.ops._C.is_gpu_compress_available())
        return op, available
    except (ImportError, AttributeError, RuntimeError):
        pass

    # Try ROCm path
    try:
        import vllm._rocm_C  # noqa: F401
        op = _torch.ops._rocm_C.compress_tensor
        available = bool(_torch.ops._rocm_C.is_gpu_compress_available())
        return op, available
    except (ImportError, AttributeError, RuntimeError):
        pass

    return None, False


class CompressedModelLoader(DefaultModelLoader):
    """
    Load model weights from a compressed checkpoint (.cbin format).

    Produced by: tools/compress_weights.py
    Format:      compression_index.json + weights_NNNNN.cbin shards
    """

    # Extra config keys understood by this loader
    _ALLOWED_EXTRA_KEYS = {
        "enable_jit_decompress",
        "prefetch_layers",
        "gpu_decompress",
        "algorithm",  # "gdeflate" (default) or "lz4"
    }

    def __init__(self, load_config: LoadConfig) -> None:
        # Bypass DefaultModelLoader's strict key check; we handle our own keys
        BaseModelLoader.__init__(self, load_config)

        extra = load_config.model_loader_extra_config or {}
        unexpected = set(extra.keys()) - self._ALLOWED_EXTRA_KEYS
        if unexpected:
            raise ValueError(
                f"Unexpected extra config keys for compressed loader: {unexpected}. "
                f"Allowed: {self._ALLOWED_EXTRA_KEYS}"
            )

        self._enable_jit = bool(extra.get("enable_jit_decompress", False))
        self._prefetch_layers = int(extra.get("prefetch_layers", 1))
        self._gpu_decompress = bool(extra.get("gpu_decompress", True))
        self._jit_algorithm = str(extra.get("algorithm", "lz4"))

        if self._jit_algorithm not in ("lz4", "gdeflate"):
            raise ValueError(
                f"Unsupported JIT algorithm: {self._jit_algorithm!r}. "
                "Supported: 'lz4', 'gdeflate'."
            )

        # Check GPU decompress availability
        self._gpu_decompress_op, self._hipcomp_available = (
            _try_get_gpu_decompress_op()
        )
        if self._gpu_decompress and not self._hipcomp_available:
            logger.info(
                "GPU decompression not available "
                "(build without nvCOMP/hipCOMP). Using CPU zlib fallback."
            )

        # Check GPU compress availability (needed for gdeflate JIT setup)
        self._gpu_compress_op, self._hipcomp_compress_available = (
            _try_get_gpu_compress_op()
        )
        if self._jit_algorithm == "gdeflate" and not self._hipcomp_compress_available:
            raise RuntimeError(
                "JIT algorithm 'gdeflate' requires GPU compression (nvCOMP/hipCOMP). "
                "Rebuild vLLM with -DVLLM_NVCOMP_PATH=... (CUDA) or "
                "-DVLLM_HIPCOMP_PATH=... (ROCm), or use algorithm='lz4'."
            )

        self._zipnn = _try_import_zipnn()

    # ------------------------------------------------------------------
    # Weight loading: Mode 1 (load-all, decompress at startup)
    # ------------------------------------------------------------------

    def get_all_weights(
        self,
        model_config: ModelConfig,
        model: torch.nn.Module,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Override: yield decompressed tensors from .cbin shards."""
        model_path = Path(model_config.model)
        index = _load_compression_index(model_path)
        yield from self._compressed_weights_iterator(model_path, index)

    def _compressed_weights_iterator(
        self,
        model_path: Path,
        index: dict,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Iterate over all compressed tensors, decompressing on CPU."""
        algorithm: str = index["algorithm"]
        tensors: dict = index["tensors"]

        # Group tensors by shard for sequential reads
        shard_to_tensors: dict[str, list[tuple[str, dict]]] = {}
        for name, meta in tensors.items():
            shard = meta["shard"]
            shard_to_tensors.setdefault(shard, []).append((name, meta))

        total = len(tensors)
        t0 = time.perf_counter()
        total_compressed = 0

        with tqdm(
            total=total,
            desc="Loading compressed weights",
            unit="tensor",
            disable=not self.load_config.use_tqdm_on_load,
        ) as pbar:
            for shard_name in sorted(shard_to_tensors):
                shard_path = model_path / shard_name
                if not shard_path.exists():
                    raise FileNotFoundError(
                        f"Shard file missing: {shard_path}"
                    )
                with shard_path.open("rb") as f:
                    for name, meta in shard_to_tensors[shard_name]:
                        f.seek(meta["byte_offset"])
                        compressed = f.read(meta["compressed_size"])
                        total_compressed += len(compressed)

                        tensor = _decompress_tensor_cpu(
                            compressed, meta, algorithm, self._zipnn
                        )
                        pbar.update(1)
                        yield name, tensor

        elapsed = time.perf_counter() - t0
        throughput = (total_compressed / 1024**2) / elapsed if elapsed > 0 else 0
        logger.info(
            "Compressed weights loaded: %d tensors in %.1fs "
            "(decompression throughput: %.0f MB/s compressed)",
            total,
            elapsed,
            throughput,
        )

    # ------------------------------------------------------------------
    # Weight loading: Mode 2 (JIT streaming)
    # Applies AFTER load_weights() completes, wraps CPU-offloaded modules.
    # ------------------------------------------------------------------

    def load_model(
        self,
        vllm_config: Any,
        model_config: ModelConfig,
        prefix: str = "",
    ) -> torch.nn.Module:
        """Load model, then optionally apply JIT compression to CPU-offloaded layers."""
        model = super().load_model(vllm_config, model_config, prefix)

        if self._enable_jit:
            logger.info(
                "JIT decompression mode enabled. "
                "Scanning CPU-offloaded modules for compression..."
            )
            self._apply_jit_compression(model, vllm_config)

        return model

    def _apply_jit_compression(
        self,
        model: torch.nn.Module,
        vllm_config: Any,
    ) -> None:
        """
        For each module whose parameters are on CPU (already offloaded),
        compress them in-place and replace the forward wrapper with one that
        decompresses before execution.
        """
        device = torch.device(vllm_config.device_config.device)

        compressed_count = 0
        for module_name, module in model.named_modules():
            params = list(module.parameters(recurse=False))
            if not params:
                continue
            if params[0].device.type != "cpu":
                continue  # not offloaded

            compressed_state = self._compress_module_cpu(module)
            if not compressed_state:
                continue

            self._install_jit_forward(module, compressed_state, device)
            compressed_count += 1

        logger.info(
            "JIT compression applied to %d CPU-offloaded module(s). "
            "Algorithm: %s  GPU decompress: %s",
            compressed_count,
            self._jit_algorithm,
            "nvCOMP/hipCOMP" if self._hipcomp_available else "CPU zlib fallback",
        )

    def _compress_module_cpu(
        self, module: torch.nn.Module
    ) -> dict[str, tuple[torch.Tensor | bytes, dict]]:
        """
        Compress all CPU parameters of a module in-place.

        Returns: {param_name: (comp_data, meta)} where:
          - For LZ4 (GPU decompress path): comp_data is a pinned CPU tensor
            (torch.Tensor, dtype=uint8, pin_memory=True).  Page-locked memory
            lets the DMA engine copy directly to GPU with a single transfer and
            no intermediate bounce buffer on the hot path.
          - For zlib (CPU fallback): comp_data is plain bytes.

        Clears the parameter data from the module to free CPU RAM.
        """
        compressed: dict[str, tuple[torch.Tensor | bytes, dict]] = {}

        for name, param in list(module.named_parameters(recurse=False)):
            tensor = param.data
            if tensor.device.type != "cpu":
                continue

            # bfloat16 and float8_* are not supported by numpy directly —
            # view as uint8 bytes first (same pattern as compress_weights.py).
            t = tensor.contiguous()
            if t.dtype in (torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2):
                raw = t.view(torch.uint8).numpy().tobytes()
            else:
                raw = t.numpy().tobytes()
            jit_algorithm = "deflate"
            comp_data: torch.Tensor | bytes

            if self._gpu_decompress and self._jit_algorithm == "gdeflate":
                # GDeflate path: compress on GPU, store result in pinned CPU RAM.
                # One-time H2D + GPU compress + D2H at setup; hot path is just
                # pinned → GPU DMA + GPU decompress (no CPU involvement).
                raw_gpu = torch.frombuffer(bytearray(raw), dtype=torch.uint8).cuda()
                comp_cpu = self._gpu_compress_op(raw_gpu, "gdeflate", 0)
                comp_data = torch.empty(
                    comp_cpu.numel(), dtype=torch.uint8, pin_memory=True
                )
                comp_data.copy_(comp_cpu)
                jit_algorithm = "gdeflate"

            elif self._gpu_decompress:
                # LZ4 path: store in pinned CPU RAM for direct DMA on hot path.
                # Prefer CPU lz4.block (no H2D+D2H overhead).  Fall back to GPU
                # LZ4 (hipCOMP) if lz4 is not installed, or to zlib as last resort.
                try:
                    comp_bytes = _compress_lz4_chunked(raw)
                    comp_data = torch.empty(
                        len(comp_bytes), dtype=torch.uint8, pin_memory=True
                    )
                    comp_data.copy_(
                        torch.frombuffer(bytearray(comp_bytes), dtype=torch.uint8)
                    )
                    jit_algorithm = "lz4"
                except ImportError:
                    if self._hipcomp_compress_available and self._gpu_compress_op is not None:
                        raw_gpu = torch.frombuffer(
                            bytearray(raw), dtype=torch.uint8
                        ).cuda()
                        comp_cpu = self._gpu_compress_op(raw_gpu, "lz4", 0)
                        comp_data = torch.empty(
                            comp_cpu.numel(), dtype=torch.uint8, pin_memory=True
                        )
                        comp_data.copy_(comp_cpu)
                        jit_algorithm = "lz4"
                    else:
                        comp_data = zlib.compress(raw, level=6)

            else:
                comp_data = zlib.compress(raw, level=6)

            meta = {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype).replace("torch.", ""),
                "original_size": len(raw),
                "algorithm": jit_algorithm,
            }
            compressed[name] = (comp_data, meta)

            # Free the decompressed CPU tensor to save RAM
            param.data = torch.empty(0, dtype=tensor.dtype, device="cpu")

        return compressed

    def _install_jit_forward(
        self,
        module: torch.nn.Module,
        compressed_state: dict[str, tuple[torch.Tensor | bytes, dict]],
        device: torch.device,
    ) -> None:
        """
        Replace module.forward with a JIT decompression wrapper.
        Mirrors the pattern from vllm/model_executor/models/utils.py:682.
        """
        from torch._functorch.functional_call import functional_call  # noqa: PLC0415

        gpu_decompress = self._gpu_decompress and self._gpu_decompress_op is not None
        gpu_decompress_op = self._gpu_decompress_op
        original_forward = module.forward

        def jit_forward(*args, **kwargs):
            module.forward = original_forward  # restore for inner calls

            if gpu_decompress:
                # GPU path: DMA from pinned CPU tensor → GPU, then GPU-decompress.
                # comp_data is a pinned uint8 tensor (set up in _compress_module_cpu),
                # so .to(device) is a single DMA transfer with no staging copy.
                # The algorithm ("lz4" or "gdeflate") is passed to the op so it
                # dispatches to the correct hipCOMP batched function.
                device_state = {}
                for name, (comp_data, meta) in compressed_state.items():
                    comp_gpu = comp_data.to(device, non_blocking=True)  # type: ignore[union-attr]
                    device_state[name] = gpu_decompress_op(
                        comp_gpu,
                        meta["shape"],
                        meta["dtype"],
                        meta["original_size"],
                        meta["algorithm"],
                    )
            else:
                # CPU fallback: decompress on CPU, non-blocking H2D copy.
                # GDeflate is GPU-only; error out if it ends up here.
                device_state = {}
                for name, (comp_data, meta) in compressed_state.items():
                    alg = meta.get("algorithm", "deflate")
                    if alg == "gdeflate":
                        raise RuntimeError(
                            "GDeflate decompression requires nvCOMP/hipCOMP (GPU path). "
                            "Enable gpu_decompress or switch to algorithm='lz4'."
                        )
                    if alg == "lz4":
                        raw = _decompress_lz4_cpu(comp_data, meta["original_size"])  # type: ignore[arg-type]
                    else:
                        raw = zlib.decompress(comp_data)  # type: ignore[arg-type]
                    cpu_tensor = _bytes_to_tensor(
                        raw, meta["shape"], meta["dtype"]
                    )
                    device_state[name] = cpu_tensor.to(device, non_blocking=True)

            output = functional_call(
                module, device_state, args=args, kwargs=kwargs, tie_weights=False
            )
            module.forward = jit_forward  # re-install for next call
            return output

        module.forward = jit_forward

    # ------------------------------------------------------------------
    # download_model: delegate entirely to parent (no compressed HF download)
    # ------------------------------------------------------------------

    def download_model(self, model_config: ModelConfig) -> None:
        super().download_model(model_config)

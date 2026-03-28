#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Offline model weight compression tool.

Compresses safetensors model checkpoints into a custom per-tensor compressed
format (.cbin shards + compression_index.json) suitable for use with
CompressedModelLoader and GPU-side decompression via nvCOMP.

Usage:
    python tools/compress_weights.py \\
        --model-path /models/llama-3-8b-gptq \\
        --output-path /models/llama-3-8b-gptq-compressed \\
        --algorithm deflate \\
        --level 6 \\
        --workers 4 \\
        --verify
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch
from safetensors.torch import load_file
from tqdm import tqdm

# Float dtypes that benefit from ZipNN byte-shuffle pre-processing
_FLOAT_DTYPES = {
    torch.float32,
    torch.float16,
    torch.bfloat16,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
}

# Max tensors per .cbin shard file (controls shard granularity)
_DEFAULT_TENSORS_PER_SHARD = 512


def _try_import_zipnn():
    try:
        import zipnn  # type: ignore[import]
        return zipnn
    except ImportError:
        return None


# Uncompressed chunk size for the nvCOMP batched LZ4 API.
# nvCOMP's recommended value is 65536 bytes (64 KB).
# This constant MUST match DECOMP_CHUNK_SIZE in weight_decompress.cu.
_LZ4_CHUNK_SIZE = 65536


def _compress_lz4(data: bytes) -> bytes:
    """Compress using chunked LZ4 (GPU-decompressable via nvCOMP batched API).

    Splits data into _LZ4_CHUNK_SIZE chunks, compresses each independently,
    and prepends a header so the GPU can dispatch one thread block per chunk.

    Format:
        [4 bytes: uint32 n_chunks]
        [n_chunks × 4 bytes: uint32 comp_size[i]]
        [chunk_0 compressed bytes]
        [chunk_1 compressed bytes]
        ...

    This matches the chunked format parsed by the nvCOMP LZ4 path in
    weight_decompress.cu.
    """
    import struct

    try:
        import lz4.block as lz4block  # type: ignore[import]
    except ImportError:
        raise ImportError(
            "lz4 is required for LZ4 compression (GPU-decompressable). "
            "Install with: pip install lz4"
        )

    chunks = []
    for offset in range(0, len(data), _LZ4_CHUNK_SIZE):
        chunk_data = data[offset : offset + _LZ4_CHUNK_SIZE]
        chunks.append(lz4block.compress(chunk_data, store_size=False))

    n_chunks = len(chunks)
    header = struct.pack(f"<I{n_chunks}I", n_chunks, *(len(c) for c in chunks))
    return header + b"".join(chunks)


def _decompress_lz4(data: bytes, original_size: int) -> bytes:
    """Decompress chunked LZ4 bytes on CPU."""
    import struct

    try:
        import lz4.block as lz4block  # type: ignore[import]
    except ImportError:
        raise ImportError(
            "lz4 is required for LZ4 decompression. "
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


def _compress_lz4_gpu(data: bytes) -> bytes:
    """Compress using LZ4 via nvCOMP GPU.

    Same chunked format as _compress_lz4() but compressed on GPU with nvCOMP
    instead of CPU lz4.block.  Useful when:
      - The lz4 Python package is not installed.
      - GPU compression throughput is needed (large tensors).
      - Called from JIT mode in CompressedModelLoader at model load time.

    Output is byte-compatible with _compress_lz4() and decompress_tensor_gpu_lz4().

    Requires vLLM built with nvCOMP and a CUDA GPU.
    """
    try:
        import vllm._C  # noqa: F401 — registers ops
    except ImportError as e:
        raise RuntimeError(
            "GPU LZ4 compression requires vLLM built with nvCOMP. "
            f"Import error: {e}"
        )
    compress_op = torch.ops._C.compress_tensor
    if not bool(torch.ops._C.is_gpu_compress_available()):
        raise RuntimeError(
            "GPU LZ4 compression requires nvCOMP. "
            "This vLLM build does not include nvCOMP support. "
            "Rebuild with -DVLLM_NVCOMP_PATH=..."
        )

    raw_tensor = torch.frombuffer(bytearray(data), dtype=torch.uint8).cuda()
    comp_cpu = compress_op(raw_tensor, "lz4", 0)
    return comp_cpu.numpy().tobytes()


def _compress_gdeflate_gpu(data: bytes, level: int = 0) -> bytes:
    """Compress using GDeflate via nvCOMP GPU (GPU-decompressable).

    Moves raw bytes to GPU, calls the nvCOMP GDeflate compress op, and
    returns bytes in the same chunked format as LZ4.

    level: GDeflate algo level:
        0 = high-throughput / fast (default)
        1 = high-compression-ratio
        2 = entropy-only (fastest, lowest ratio)

    Requires vLLM built with nvCOMP and a CUDA GPU.
    """
    try:
        import vllm._C  # noqa: F401 — registers ops
    except ImportError as e:
        raise RuntimeError(
            "GDeflate compression requires vLLM built with nvCOMP. "
            f"Import error: {e}"
        )
    compress_op = torch.ops._C.compress_tensor
    if not bool(torch.ops._C.is_gpu_compress_available()):
        raise RuntimeError(
            "GDeflate compression requires nvCOMP. "
            "This vLLM build does not include nvCOMP support. "
            "Rebuild with -DVLLM_NVCOMP_PATH=..."
        )

    # GDeflate algo level must be 0-2. Clamp in case a deflate-style level (1-9)
    # is passed from the CLI (--level default is 6).
    clamped_level = min(level, 2)

    raw_tensor = torch.frombuffer(bytearray(data), dtype=torch.uint8).cuda()
    comp_cpu = compress_op(raw_tensor, "gdeflate", clamped_level)
    return comp_cpu.numpy().tobytes()


def _decompress_gdeflate_gpu(data: bytes, shape: list, dtype_str: str,
                              original_size: int) -> torch.Tensor:
    """Decompress GDeflate bytes on GPU (for --verify pass)."""
    try:
        import vllm._C  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            f"GDeflate decompression requires nvCOMP: {e}"
        )
    decompress_op = torch.ops._C.decompress_tensor
    comp_tensor = torch.frombuffer(bytearray(data), dtype=torch.uint8).cuda()
    out_gpu = decompress_op(comp_tensor, shape, dtype_str, original_size,
                            "gdeflate")
    return out_gpu.cpu()


def _compress_deflate(data: bytes, level: int) -> bytes:
    """Compress using DEFLATE (zlib). CPU decompression only."""
    return zlib.compress(data, level=level)


def _compress_zstd(data: bytes, level: int) -> bytes:
    """Compress using Zstd. CPU decompression only."""
    try:
        import zstandard as zstd  # type: ignore[import]
    except ImportError:
        raise ImportError(
            "zstandard is required for Zstd compression. "
            "Install with: pip install zstandard"
        )
    cctx = zstd.ZstdCompressor(level=level)
    return cctx.compress(data)


def _decompress_deflate(data: bytes) -> bytes:
    return zlib.decompress(data)


def _decompress_zstd(data: bytes) -> bytes:
    try:
        import zstandard as zstd  # type: ignore[import]
    except ImportError:
        raise ImportError("zstandard is required for Zstd decompression.")
    dctx = zstd.ZstdDecompressor()
    return dctx.decompress(data)


def _tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    """Convert tensor to raw bytes (C-contiguous).

    For dtypes that numpy does not natively support (bfloat16, float8_*),
    we reinterpret the storage as uint8 before converting.
    """
    import numpy as np

    t = tensor.contiguous()
    # dtypes unsupported by numpy: view as uint8 first
    if t.dtype in (torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2):
        return t.view(torch.uint8).numpy().tobytes()
    return t.numpy().tobytes()


def _bytes_to_tensor(
    raw: bytes, shape: list[int], dtype_str: str
) -> torch.Tensor:
    """Reconstruct tensor from raw bytes."""
    import numpy as np

    dtype_map = {
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
    torch_dtype = dtype_map.get(dtype_str)
    if torch_dtype is None:
        raise ValueError(f"Unsupported dtype in index: {dtype_str}")

    # For dtypes not directly supported by numpy, decode via uint8
    if dtype_str in ("bfloat16", "float8_e4m3fn", "float8_e5m2"):
        arr = np.frombuffer(raw, dtype=np.uint8)
        t = torch.from_numpy(arr.copy()).view(torch_dtype)
    else:
        np_dtype_map = {
            "float32": np.float32,
            "float16": np.float16,
            "int32": np.int32,
            "int8": np.int8,
            "uint8": np.uint8,
            "int64": np.int64,
        }
        arr = np.frombuffer(raw, dtype=np_dtype_map[dtype_str])
        t = torch.from_numpy(arr.copy())

    return t.reshape(shape)


def _dtype_str(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def compress_tensor(
    tensor: torch.Tensor,
    algorithm: str,
    level: int,
    zipnn_preshuffle: bool,
    zipnn_module=None,
    gpu_compress: bool = False,
) -> tuple[bytes, bool]:
    """
    Compress a single tensor.

    gpu_compress: when True and algorithm='lz4', use nvCOMP GPU compression
                  instead of CPU lz4.block (faster for large tensors, no lz4 dep).

    Returns:
        (compressed_bytes, was_preshuffled)
    """
    raw = _tensor_to_bytes(tensor)
    preshuffled = False

    if zipnn_preshuffle and zipnn_module is not None and tensor.dtype in _FLOAT_DTYPES:
        try:
            compressed = zipnn_module.compress(raw, dtype=_dtype_str(tensor.dtype))
            return compressed, True
        except Exception:
            pass  # Fall through to standard compression

    if algorithm == "lz4":
        if gpu_compress:
            compressed = _compress_lz4_gpu(raw)
        else:
            compressed = _compress_lz4(raw)
    elif algorithm == "gdeflate":
        compressed = _compress_gdeflate_gpu(raw, level)
    elif algorithm == "deflate":
        compressed = _compress_deflate(raw, level)
    elif algorithm == "zstd":
        compressed = _compress_zstd(raw, level)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm!r}")

    return compressed, preshuffled


def decompress_tensor(
    compressed: bytes,
    shape: list[int],
    dtype_str: str,
    algorithm: str,
    preshuffled: bool,
    zipnn_module=None,
) -> torch.Tensor:
    """Decompress a tensor from bytes."""
    # Compute original byte size from shape + dtype for LZ4 (needs uncompressed_size)
    _elem_sizes = {
        "float32": 4, "float16": 2, "bfloat16": 2,
        "int32": 4, "int8": 1, "uint8": 1, "int64": 8,
        "float8_e4m3fn": 1, "float8_e5m2": 1,
    }
    numel = 1
    for s in shape:
        numel *= s
    original_size = numel * _elem_sizes[dtype_str]

    # GDeflate has no CPU decompressor — use GPU for verification
    if algorithm == "gdeflate":
        return _decompress_gdeflate_gpu(compressed, shape, dtype_str, original_size)

    if preshuffled and zipnn_module is not None:
        try:
            raw = zipnn_module.decompress(compressed)
        except Exception:
            if algorithm == "lz4":
                raw = _decompress_lz4(compressed, original_size)
            elif algorithm == "deflate":
                raw = _decompress_deflate(compressed)
            else:
                raw = _decompress_zstd(compressed)
    elif algorithm == "lz4":
        raw = _decompress_lz4(compressed, original_size)
    elif algorithm == "deflate":
        raw = _decompress_deflate(compressed)
    elif algorithm == "zstd":
        raw = _decompress_zstd(compressed)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm!r}")

    return _bytes_to_tensor(raw, shape, dtype_str)


def _tensor_checksum(tensor: torch.Tensor) -> str:
    return hashlib.sha256(_tensor_to_bytes(tensor)).hexdigest()[:16]


def compress_model(
    model_path: Path,
    output_path: Path,
    algorithm: str = "gdeflate",
    level: int = 6,
    zipnn_preshuffle: bool = False,
    workers: int = 4,
    tensors_per_shard: int = _DEFAULT_TENSORS_PER_SHARD,
    verify: bool = False,
    gpu_compress: bool = False,
) -> dict:
    """
    Compress a safetensors model checkpoint to the .cbin format.

    Returns:
        Summary statistics dict.
    """
    output_path.mkdir(parents=True, exist_ok=True)

    zipnn_module = _try_import_zipnn() if zipnn_preshuffle else None
    if zipnn_preshuffle and zipnn_module is None:
        print(
            "Warning: --zipnn-preshuffle requested but zipnn is not installed. "
            "Falling back to plain compression. Install with: pip install zipnn"
        )

    # Gather all safetensors files
    st_files = sorted(model_path.glob("*.safetensors"))
    if not st_files:
        raise FileNotFoundError(
            f"No .safetensors files found in {model_path}. "
            "Ensure the model path contains a HuggingFace checkpoint."
        )

    print(f"Found {len(st_files)} safetensors shard(s) in {model_path}")

    # Copy non-tensor files verbatim
    for f in model_path.iterdir():
        if f.suffix not in (".safetensors",) and f.is_file():
            shutil.copy2(f, output_path / f.name)
    print(f"Copied config/tokenizer files to {output_path}")

    # Load all tensors from all shards (preserving order)
    # We collect (name, tensor) pairs then assign to shards
    all_tensors: list[tuple[str, torch.Tensor]] = []
    print("Loading tensors from safetensors shards...")
    for st_file in tqdm(st_files, desc="Loading shards"):
        state_dict = load_file(str(st_file), device="cpu")
        for name, tensor in state_dict.items():
            all_tensors.append((name, tensor))
    print(f"Loaded {len(all_tensors)} tensors total")

    # Assign tensors to .cbin shards
    shard_assignments: list[list[tuple[str, torch.Tensor]]] = []
    for i in range(0, len(all_tensors), tensors_per_shard):
        shard_assignments.append(all_tensors[i : i + tensors_per_shard])
    print(f"Organized into {len(shard_assignments)} .cbin shard(s)")

    # Build index and compress
    index: dict = {
        "algorithm": algorithm,
        "zipnn_preshuffle": zipnn_preshuffle and zipnn_module is not None,
        "level": level,
        "tensors": {},
    }
    total_original = 0
    total_compressed = 0
    dtype_stats: dict[str, dict] = {}

    # GPU compression (LZ4 --gpu-compress or GDeflate) serialises all concurrent
    # calls — multi-threading only adds CUDA context overhead.
    gpu_compress_active = gpu_compress or algorithm == "gdeflate"
    effective_workers = 1 if gpu_compress_active else workers
    if gpu_compress_active and workers > 1:
        alg_label = "GDeflate" if algorithm == "gdeflate" else "LZ4 GPU"
        print(f"Note: {alg_label} uses GPU compression; forcing single-threaded "
              "(GPU serialises concurrent calls).")

    t_start = time.perf_counter()

    for shard_idx, shard_tensors in enumerate(
        tqdm(shard_assignments, desc="Compressing shards")
    ):
        shard_name = f"weights_{shard_idx + 1:05d}.cbin"
        shard_path = output_path / shard_name
        byte_offset = 0

        def _compress_one(args):
            name, tensor = args
            compressed, preshuffled = compress_tensor(
                tensor, algorithm, level, zipnn_preshuffle, zipnn_module,
                gpu_compress=gpu_compress_active,
            )
            return name, tensor, compressed, preshuffled

        with shard_path.open("wb") as shard_file:
            if effective_workers > 1:
                with ThreadPoolExecutor(max_workers=effective_workers) as pool:
                    futures = {
                        pool.submit(_compress_one, item): item[0]
                        for item in shard_tensors
                    }
                    # Preserve original order for byte_offset consistency
                    results_map = {}
                    for fut in as_completed(futures):
                        name, tensor, compressed, preshuffled = fut.result()
                        results_map[name] = (tensor, compressed, preshuffled)
                    ordered_results = [
                        (name, *results_map[name]) for name, _ in shard_tensors
                    ]
            else:
                ordered_results = [
                    (name, tensor, *compress_tensor(
                        tensor, algorithm, level, zipnn_preshuffle, zipnn_module
                    )[0:2])
                    for name, tensor in shard_tensors
                ]
                # Re-structure: compress returns (bytes, bool)
                ordered_results_fixed = []
                for name, tensor in shard_tensors:
                    compressed, preshuffled = compress_tensor(
                        tensor, algorithm, level, zipnn_preshuffle, zipnn_module,
                        gpu_compress=gpu_compress_active,
                    )
                    ordered_results_fixed.append((name, tensor, compressed, preshuffled))
                ordered_results = ordered_results_fixed

            for name, tensor, compressed, preshuffled in ordered_results:
                original_size = tensor.numel() * tensor.element_size()
                compressed_size = len(compressed)
                dtype_s = _dtype_str(tensor.dtype)

                shard_file.write(compressed)

                index["tensors"][name] = {
                    "shard": shard_name,
                    "byte_offset": byte_offset,
                    "compressed_size": compressed_size,
                    "original_size": original_size,
                    "shape": list(tensor.shape),
                    "dtype": dtype_s,
                    "preshuffled": preshuffled,
                    "compression_ratio": compressed_size / original_size,
                }

                byte_offset += compressed_size
                total_original += original_size
                total_compressed += compressed_size

                if dtype_s not in dtype_stats:
                    dtype_stats[dtype_s] = {"count": 0, "orig": 0, "comp": 0}
                dtype_stats[dtype_s]["count"] += 1
                dtype_stats[dtype_s]["orig"] += original_size
                dtype_stats[dtype_s]["comp"] += compressed_size

    elapsed = time.perf_counter() - t_start

    # Finalize index
    index["total_original_bytes"] = total_original
    index["total_compressed_bytes"] = total_compressed
    index["dtype_stats"] = {
        k: {
            "count": v["count"],
            "original_bytes": v["orig"],
            "compressed_bytes": v["comp"],
            "ratio": v["comp"] / v["orig"] if v["orig"] > 0 else 1.0,
        }
        for k, v in dtype_stats.items()
    }

    index_path = output_path / "compression_index.json"
    with index_path.open("w") as f:
        json.dump(index, f, indent=2)

    # Verification pass
    if verify:
        print("\nVerifying round-trip correctness...")
        errors = 0
        for name, meta in tqdm(index["tensors"].items(), desc="Verifying"):
            shard_path = output_path / meta["shard"]
            with shard_path.open("rb") as f:
                f.seek(meta["byte_offset"])
                compressed = f.read(meta["compressed_size"])

            reconstructed = decompress_tensor(
                compressed,
                meta["shape"],
                meta["dtype"],
                algorithm,
                meta["preshuffled"],
                zipnn_module,
            )

            # Find original tensor
            orig_tensor = next(t for n, t in all_tensors if n == name)

            orig_check = _tensor_checksum(orig_tensor)
            rec_check = _tensor_checksum(reconstructed)
            if orig_check != rec_check:
                print(f"  MISMATCH: {name} (orig={orig_check}, got={rec_check})")
                errors += 1

        if errors == 0:
            print(f"  All {len(index['tensors'])} tensors verified OK.")
        else:
            print(f"  {errors} tensor(s) failed verification!")
            sys.exit(1)

    # Summary
    overall_ratio = total_compressed / total_original if total_original > 0 else 1.0
    throughput_mbs = (total_original / 1024 / 1024) / elapsed if elapsed > 0 else 0

    summary = {
        "algorithm": algorithm,
        "level": level,
        "zipnn_preshuffle": zipnn_preshuffle and zipnn_module is not None,
        "num_tensors": len(index["tensors"]),
        "num_shards": len(shard_assignments),
        "total_original_gb": total_original / 1024**3,
        "total_compressed_gb": total_compressed / 1024**3,
        "overall_ratio": overall_ratio,
        "reduction_pct": (1 - overall_ratio) * 100,
        "compress_throughput_mbs": throughput_mbs,
        "elapsed_s": elapsed,
        "dtype_stats": index["dtype_stats"],
    }

    return summary


def _print_summary(summary: dict) -> None:
    print("\n" + "=" * 60)
    print("Compression Summary")
    print("=" * 60)
    print(f"Algorithm:          {summary['algorithm']} (level={summary['level']})")
    if summary["zipnn_preshuffle"]:
        print("ZipNN pre-shuffle:  enabled (float tensors)")
    print(
        f"Original size:      {summary['total_original_gb']:.2f} GB"
    )
    print(
        f"Compressed size:    {summary['total_compressed_gb']:.2f} GB"
        f"  ({summary['reduction_pct']:.1f}% reduction)"
    )
    print(f"Overall ratio:      {summary['overall_ratio']:.3f}")
    print(f"Throughput:         {summary['compress_throughput_mbs']:.0f} MB/s")
    print(f"Elapsed:            {summary['elapsed_s']:.1f}s")
    print(f"Tensors:            {summary['num_tensors']} in {summary['num_shards']} shard(s)")

    print("\nBreakdown by dtype:")
    for dtype, stats in sorted(summary["dtype_stats"].items()):
        orig_gb = stats["original_bytes"] / 1024**3
        comp_gb = stats["compressed_bytes"] / 1024**3
        pct = (1 - stats["ratio"]) * 100
        print(
            f"  {dtype:20s}  {stats['count']:5d} tensors  "
            f"{orig_gb:.3f} GB → {comp_gb:.3f} GB  (-{pct:.1f}%)"
        )
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Compress vLLM model weights for use with CompressedModelLoader.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic DEFLATE compression (default)
  python tools/compress_weights.py \\
      --model-path /models/llama-3-8b-gptq \\
      --output-path /models/llama-3-8b-gptq-c

  # Zstd with ZipNN pre-shuffle for float tensors
  python tools/compress_weights.py \\
      --model-path /models/llama-3-8b \\
      --output-path /models/llama-3-8b-c \\
      --algorithm zstd --level 3 --zipnn-preshuffle --verify

  # Then load with vLLM:
  vllm serve /models/llama-3-8b-gptq-c --load-format compressed
        """,
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to the source model directory (contains .safetensors files).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Destination directory for compressed model.",
    )
    parser.add_argument(
        "--algorithm",
        choices=["gdeflate", "lz4", "deflate", "zstd"],
        default="gdeflate",
        help="Compression algorithm. "
             "'gdeflate' = GDeflate via nvCOMP GPU (default; GPU-decompressable; "
             "requires nvCOMP + CUDA GPU; -DVLLM_NVCOMP_PATH=... at build time). "
             "'lz4' = LZ4 chunked (GPU-decompressable via nvCOMP; pip install lz4). "
             "'deflate' = zlib DEFLATE (CPU decompression only). "
             "'zstd' = Zstandard (CPU only; pip install zstandard).",
    )
    parser.add_argument(
        "--gpu-compress",
        action="store_true",
        default=False,
        help="Use GPU (nvCOMP) for compression instead of CPU. "
             "Applies to --algorithm lz4: replaces the lz4 Python package with the "
             "nvCOMP LZ4 GPU compressor, which is faster for large tensors and "
             "requires no extra pip packages. "
             "GDeflate always uses GPU compression regardless of this flag.",
    )
    parser.add_argument(
        "--level",
        type=int,
        default=6,
        help="Compression level. For deflate: 1 (fast) to 9 (best). "
             "For zstd: 1 (fast) to 22 (best). Default: 6.",
    )
    parser.add_argument(
        "--zipnn-preshuffle",
        action="store_true",
        default=False,
        help="Apply ZipNN byte-shuffle pre-processing for float tensors before compression. "
             "Improves ratio ~17%% on FP16/BF16. Requires 'pip install zipnn'.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel compression workers. Default: 4.",
    )
    parser.add_argument(
        "--tensors-per-shard",
        type=int,
        default=_DEFAULT_TENSORS_PER_SHARD,
        help=f"Number of tensors per .cbin shard file. Default: {_DEFAULT_TENSORS_PER_SHARD}.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        default=False,
        help="After compression, verify each tensor decompresses to match the original "
             "(SHA-256 check). Doubles runtime.",
    )
    args = parser.parse_args()

    if not args.model_path.exists():
        parser.error(f"Model path does not exist: {args.model_path}")
    if args.output_path == args.model_path:
        parser.error("--output-path must differ from --model-path")

    print(f"Compressing {args.model_path} → {args.output_path}")
    print(f"Algorithm: {args.algorithm} (level={args.level})")

    summary = compress_model(
        model_path=args.model_path,
        output_path=args.output_path,
        algorithm=args.algorithm,
        level=args.level,
        zipnn_preshuffle=args.zipnn_preshuffle,
        workers=args.workers,
        tensors_per_shard=args.tensors_per_shard,
        verify=args.verify,
        gpu_compress=args.gpu_compress,
    )
    _print_summary(summary)

    print(f"\nCompressed model saved to: {args.output_path}")
    print("Load with: vllm serve <output-path> --load-format compressed")


if __name__ == "__main__":
    main()

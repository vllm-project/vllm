#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark suite for model weight compression.

Measures five things:
  1. Offline compression ratio and throughput (per dtype)
  2. Model loading time: baseline vs compressed loader
  3. Peak memory footprint: GPU VRAM and CPU RAM
  4. Inference throughput: tok/s across decompression backends
  5. Decompression microbenchmark: GB/s for ops.decompress_tensor

Usage:
    # Benchmark compression ratio only (no GPU needed)
    python benchmarks/benchmark_compression.py \\
        --model /models/llama-3-8b-gptq \\
        --sections compression

    # Full benchmark (requires GPU)
    python benchmarks/benchmark_compression.py \\
        --model /models/llama-3-8b-gptq \\
        --backends baseline compressed-load-all jit-cpu \\
        --num-inference-steps 50 \\
        --output-json results/compression_bench.json

    # Microbenchmark: decompression speed only
    python benchmarks/benchmark_compression.py \\
        --model /models/llama-3-8b-gptq \\
        --sections decompress-microbench
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
import zlib
from pathlib import Path
from typing import Any

import torch

# Dtypes numpy can't handle directly — must view as uint8 first
_NUMPY_UNSUPPORTED_DTYPES = {torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2}


def _tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    t = tensor.contiguous()
    if t.dtype in _NUMPY_UNSUPPORTED_DTYPES:
        return t.view(torch.uint8).numpy().tobytes()
    return t.numpy().tobytes()


# ---------------------------------------------------------------------------
# Compression helpers (GPU-based for gdeflate/lz4, CPU for deflate/zstd)
# ---------------------------------------------------------------------------

def _compress_bytes(raw: bytes, algorithm: str, level: int) -> bytes:
    """Compress raw bytes using the given algorithm.

    For gdeflate and lz4, uses the nvCOMP GPU ops (no CPU implementation).
    For deflate and zstd, uses CPU libraries.
    """
    if algorithm == "deflate":
        return zlib.compress(raw, level=level)
    elif algorithm == "zstd":
        import zstandard as zstd
        return zstd.ZstdCompressor(level=level).compress(raw)
    elif algorithm in ("gdeflate", "lz4"):
        try:
            import vllm._C  # noqa: F401 — registers ops
        except ImportError:
            raise RuntimeError(
                f"{algorithm} compression requires vLLM built with nvCOMP."
            )
        if not bool(torch.ops._C.is_gpu_compress_available()):
            raise RuntimeError(
                f"{algorithm} compression requires nvCOMP. "
                "Rebuild vLLM with -DVLLM_NVCOMP_PATH=..."
            )
        # GDeflate algo level: 0=fast, 1=high-ratio, 2=entropy-only (must be 0-2).
        # LZ4 ignores level entirely. Clamp so deflate-style levels (1-9) don't error.
        gpu_level = min(level, 2) if algorithm == "gdeflate" else 0

        # The GPU pool supports at most MAX_POOL_CHUNKS=16384 chunks of 65536 bytes
        # = 1 GB per call. Split larger tensors into 1 GB segments and compress
        # each independently (ratio measurement remains accurate).
        _MAX_SEGMENT = 16384 * 65536  # 1 GB
        if len(raw) <= _MAX_SEGMENT:
            raw_gpu = torch.frombuffer(bytearray(raw), dtype=torch.uint8).cuda()
            comp_cpu = torch.ops._C.compress_tensor(raw_gpu, algorithm, gpu_level)
            return comp_cpu.numpy().tobytes()

        segments = []
        for off in range(0, len(raw), _MAX_SEGMENT):
            seg = raw[off : off + _MAX_SEGMENT]
            seg_gpu = torch.frombuffer(bytearray(seg), dtype=torch.uint8).cuda()
            seg_comp = torch.ops._C.compress_tensor(seg_gpu, algorithm, gpu_level)
            segments.append(seg_comp.numpy().tobytes())
        return b"".join(segments)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


# ---------------------------------------------------------------------------
# Section 1: Offline compression ratio + throughput
# ---------------------------------------------------------------------------

def bench_compression_ratio(
    model_path: Path,
    algorithm: str = "deflate",
    level: int = 6,
) -> dict:
    """
    Compress model tensors in-memory (no disk output) and measure ratios.
    For gdeflate/lz4, requires a CUDA GPU with nvCOMP.
    """
    try:
        from safetensors.torch import load_file
    except ImportError:
        print("ERROR: safetensors is required. pip install safetensors")
        sys.exit(1)

    st_files = sorted(model_path.glob("*.safetensors"))
    if not st_files:
        print(f"ERROR: No .safetensors files found in {model_path}")
        sys.exit(1)

    gpu_algo = algorithm in ("gdeflate", "lz4")
    print(f"\n{'='*60}")
    print(f"Section 1: Compression Ratio — {algorithm} (level={level})"
          f"{' [GPU]' if gpu_algo else ' [CPU]'}")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"Shards: {len(st_files)}")

    dtype_stats: dict[str, dict] = {}
    total_orig = 0
    total_comp = 0
    t0 = time.perf_counter()

    for st_file in st_files:
        sd = load_file(str(st_file), device="cpu")
        for name, tensor in sd.items():
            raw = _tensor_to_bytes(tensor)
            comp = _compress_bytes(raw, algorithm, level)

            dtype_s = str(tensor.dtype).replace("torch.", "")
            if dtype_s not in dtype_stats:
                dtype_stats[dtype_s] = {
                    "count": 0, "orig_bytes": 0, "comp_bytes": 0
                }
            dtype_stats[dtype_s]["count"] += 1
            dtype_stats[dtype_s]["orig_bytes"] += len(raw)
            dtype_stats[dtype_s]["comp_bytes"] += len(comp)
            total_orig += len(raw)
            total_comp += len(comp)

    elapsed = time.perf_counter() - t0
    throughput = (total_orig / 1024**2) / elapsed if elapsed > 0 else 0
    overall_ratio = total_comp / total_orig if total_orig > 0 else 1.0

    print(f"\nTotal original:   {total_orig / 1024**3:.2f} GB")
    print(f"Total compressed: {total_comp / 1024**3:.2f} GB")
    print(f"Overall ratio:    {overall_ratio:.3f}  ({(1-overall_ratio)*100:.1f}% reduction)")
    print(f"Throughput:       {throughput:.0f} MB/s")
    print(f"Elapsed:          {elapsed:.1f}s")

    print("\nBy dtype:")
    for dtype, s in sorted(dtype_stats.items()):
        ratio = s["comp_bytes"] / s["orig_bytes"] if s["orig_bytes"] > 0 else 1.0
        print(
            f"  {dtype:20s}  {s['count']:5d} tensors  "
            f"{s['orig_bytes']/1024**3:.3f} GB → {s['comp_bytes']/1024**3:.3f} GB  "
            f"(-{(1-ratio)*100:.1f}%)"
        )

    return {
        "algorithm": algorithm,
        "level": level,
        "total_original_gb": total_orig / 1024**3,
        "total_compressed_gb": total_comp / 1024**3,
        "overall_ratio": overall_ratio,
        "reduction_pct": (1 - overall_ratio) * 100,
        "throughput_mbs": throughput,
        "elapsed_s": elapsed,
        "dtype_stats": {
            k: {
                "count": v["count"],
                "ratio": v["comp_bytes"] / v["orig_bytes"],
                "reduction_pct": (1 - v["comp_bytes"] / v["orig_bytes"]) * 100,
            }
            for k, v in dtype_stats.items()
        },
    }


# ---------------------------------------------------------------------------
# Section 2: Loading time comparison
# ---------------------------------------------------------------------------

def _make_compressed_model(model_path: Path, out_path: Path, algorithm: str, level: int):
    """Helper: run compress_weights.py programmatically."""
    tools_dir = Path(__file__).parent.parent / "tools"
    compress_script = tools_dir / "compress_weights.py"
    if not compress_script.exists():
        raise FileNotFoundError(f"compress_weights.py not found at {compress_script}")

    import importlib.util
    spec = importlib.util.spec_from_file_location("compress_weights", compress_script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    gpu_compress = algorithm in ("gdeflate", "lz4")
    mod.compress_model(
        model_path=model_path,
        output_path=out_path,
        algorithm=algorithm,
        level=level,
        zipnn_preshuffle=False,
        workers=4,
        verify=False,
        gpu_compress=gpu_compress,
    )


def bench_loading_time(
    model_path: Path,
    compressed_path: Path,
    dtype: str = "auto",
) -> dict:
    """Measure loading time: DefaultModelLoader vs CompressedModelLoader."""
    print(f"\n{'='*60}")
    print("Section 2: Loading Time Comparison")
    print(f"{'='*60}")

    results = {}

    # Baseline
    print("\nLoading baseline (DefaultModelLoader)...")
    t0 = time.perf_counter()
    try:
        from vllm import LLM
        llm = LLM(
            model=str(model_path),
            dtype=dtype,
            gpu_memory_utilization=0.85,
            enforce_eager=True,
        )
        baseline_s = time.perf_counter() - t0
        del llm
        torch.cuda.empty_cache()
        results["baseline_load_s"] = baseline_s
        print(f"  Baseline:   {baseline_s:.1f}s")
    except Exception as e:
        print(f"  Baseline load failed: {e}")
        results["baseline_load_s"] = None

    # Compressed
    print("\nLoading compressed (CompressedModelLoader)...")
    t0 = time.perf_counter()
    try:
        from vllm import LLM
        llm = LLM(
            model=str(compressed_path),
            load_format="compressed",
            dtype=dtype,
            gpu_memory_utilization=0.85,
            enforce_eager=True,
        )
        compressed_s = time.perf_counter() - t0
        del llm
        torch.cuda.empty_cache()
        results["compressed_load_s"] = compressed_s
        print(f"  Compressed: {compressed_s:.1f}s")
    except Exception as e:
        print(f"  Compressed load failed: {e}")
        results["compressed_load_s"] = None

    if results.get("baseline_load_s") and results.get("compressed_load_s"):
        diff = results["compressed_load_s"] - results["baseline_load_s"]
        print(
            f"\n  Overhead: {diff:+.1f}s "
            f"({'slower' if diff > 0 else 'faster'} than baseline)"
        )
        results["overhead_s"] = diff

    return results


# ---------------------------------------------------------------------------
# Section 3: Peak memory footprint
# ---------------------------------------------------------------------------

def bench_memory_footprint(
    model_path: Path,
    compressed_path: Path,
    cpu_offload_gb: float = 0.0,
    dtype: str = "auto",
) -> dict:
    """Measure peak GPU VRAM and CPU RAM during loading."""
    import psutil

    print(f"\n{'='*60}")
    print("Section 3: Memory Footprint")
    print(f"{'='*60}")

    results = {}
    process = psutil.Process(os.getpid())

    def _get_memory():
        gpu_mb = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        cpu_mb = process.memory_info().rss / 1024**2
        return gpu_mb, cpu_mb

    for label, model_dir, load_format, extra_config in [
        ("baseline", model_path, "auto", None),
        ("compressed_load_all", compressed_path, "compressed", None),
        ("compressed_jit_cpu", compressed_path, "compressed",
         '{"enable_jit_decompress":true,"gpu_decompress":false}'),
        ("compressed_jit_gpu", compressed_path, "compressed",
         '{"enable_jit_decompress":true,"gpu_decompress":true}'),
    ]:
        if not model_dir.exists():
            results[label] = {"error": f"{model_dir} not found"}
            continue

        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            from vllm import LLM
            kwargs: dict[str, Any] = dict(
                model=str(model_dir),
                load_format=load_format,
                dtype=dtype,
                gpu_memory_utilization=0.85,
                enforce_eager=True,
            )
            if cpu_offload_gb > 0:
                kwargs["cpu_offload_gb"] = cpu_offload_gb
            if extra_config:
                kwargs["model_loader_extra_config"] = json.loads(extra_config)

            llm = LLM(**kwargs)
            gpu_mb, cpu_mb = _get_memory()
            del llm
            torch.cuda.empty_cache()

            results[label] = {
                "peak_gpu_vram_gb": gpu_mb / 1024,
                "cpu_ram_gb": cpu_mb / 1024,
            }
            print(
                f"  {label:30s}  GPU: {gpu_mb/1024:.2f} GB  CPU: {cpu_mb/1024:.2f} GB"
            )
        except Exception as e:
            results[label] = {"error": str(e)}
            print(f"  {label:30s}  ERROR: {e}")

    return results


# ---------------------------------------------------------------------------
# Section 4: Inference throughput
# ---------------------------------------------------------------------------

def bench_inference_throughput(
    model_path: Path,
    compressed_path: Path,
    backends: list[str],
    num_warmup: int = 5,
    num_steps: int = 50,
    prompt: str = "The quick brown fox jumps over the lazy dog. ",
    max_tokens: int = 128,
    dtype: str = "auto",
) -> dict:
    """Measure tokens/second across decompression backends."""
    from vllm import LLM, SamplingParams

    print(f"\n{'='*60}")
    print("Section 4: Inference Throughput")
    print(f"{'='*60}")

    prompts = [prompt] * 8  # batch of 8

    backend_configs = {
        "baseline": dict(
            model=str(model_path), load_format="auto", dtype=dtype,
            gpu_memory_utilization=0.85, enforce_eager=True
        ),
        "compressed-load-all": dict(
            model=str(compressed_path), load_format="compressed", dtype=dtype,
            gpu_memory_utilization=0.85, enforce_eager=True
        ),
        "jit-cpu": dict(
            model=str(compressed_path), load_format="compressed", dtype=dtype,
            gpu_memory_utilization=0.5, enforce_eager=True,
            model_loader_extra_config={
                "enable_jit_decompress": True, "gpu_decompress": False
            },
        ),
        "jit-gpu": dict(
            model=str(compressed_path), load_format="compressed", dtype=dtype,
            gpu_memory_utilization=0.5, enforce_eager=True,
            model_loader_extra_config={
                "enable_jit_decompress": True, "gpu_decompress": True
            },
        ),
    }

    results = {}
    sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0.0)

    for backend in backends:
        if backend not in backend_configs:
            print(f"  Unknown backend {backend!r}, skipping.")
            continue

        model_dir = compressed_path if backend != "baseline" else model_path
        if not model_dir.exists():
            results[backend] = {"error": f"Model dir not found: {model_dir}"}
            continue

        print(f"\n  Backend: {backend}")
        try:
            llm = LLM(**backend_configs[backend])

            # Warmup
            for _ in range(num_warmup):
                llm.generate(prompts, sampling_params)

            # Measure
            t0 = time.perf_counter()
            total_tokens = 0
            for _ in range(num_steps):
                outputs = llm.generate(prompts, sampling_params)
                total_tokens += sum(
                    len(o.outputs[0].token_ids) for o in outputs
                )
            elapsed = time.perf_counter() - t0

            tok_per_s = total_tokens / elapsed
            results[backend] = {
                "tokens_per_s": tok_per_s,
                "total_tokens": total_tokens,
                "elapsed_s": elapsed,
            }
            print(f"    {tok_per_s:.1f} tok/s")

            del llm
            torch.cuda.empty_cache()

        except Exception as e:
            results[backend] = {"error": str(e)}
            print(f"    ERROR: {e}")

    # Print comparison
    if "baseline" in results and results["baseline"].get("tokens_per_s"):
        base = results["baseline"]["tokens_per_s"]
        print("\n  Comparison vs baseline:")
        for bk, r in results.items():
            if bk == "baseline" or "error" in r:
                continue
            diff = (r["tokens_per_s"] - base) / base * 100
            print(f"    {bk:30s}  {r['tokens_per_s']:.1f} tok/s  ({diff:+.1f}%)")

    return results


# ---------------------------------------------------------------------------
# Section 5: GPU decompression microbenchmark
# ---------------------------------------------------------------------------

def bench_decompress_microbench(
    model_path: Path,
    algorithm: str = "deflate",
    level: int = 6,
    iterations: int = 100,
) -> dict:
    """
    Isolate decompression speed: compress a representative tensor,
    transfer to GPU, and time repeated decompression via ops.decompress_tensor.
    Falls back to CPU timing if nvCOMP is not available.

    For gdeflate/lz4: compression uses the GPU nvCOMP op (no CPU compressor).
    CPU decompression benchmark is skipped for gdeflate (GPU-only algorithm).
    """
    from safetensors.torch import load_file

    print(f"\n{'='*60}")
    print(f"Section 5: Decompression Microbenchmark — {algorithm}")
    print(f"{'='*60}")

    # Load one representative large tensor
    st_files = sorted(model_path.glob("*.safetensors"))
    if not st_files:
        print("No safetensors files found, skipping microbenchmark.")
        return {}

    sd = load_file(str(st_files[0]), device="cpu")
    # Pick the largest tensor as representative workload
    name, tensor = max(sd.items(), key=lambda kv: kv[1].numel())
    print(f"\nRepresentative tensor: {name}  shape={list(tensor.shape)}  dtype={tensor.dtype}")

    raw = _tensor_to_bytes(tensor)
    comp = _compress_bytes(raw, algorithm, level)

    ratio = len(comp) / len(raw)
    print(f"Compressed: {len(raw)/1024**2:.1f} MB → {len(comp)/1024**2:.1f} MB (ratio={ratio:.3f})")

    results: dict[str, Any] = {
        "algorithm": algorithm,
        "tensor_name": name,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "original_mb": len(raw) / 1024**2,
        "compressed_mb": len(comp) / 1024**2,
        "ratio": ratio,
    }

    # CPU decompression benchmark (only for algorithms with a CPU decompressor)
    cpu_throughput = None
    if algorithm == "deflate":
        print(f"\n  CPU decompression ({iterations} iterations):")
        t0 = time.perf_counter()
        for _ in range(iterations):
            zlib.decompress(comp)
        cpu_elapsed = time.perf_counter() - t0
        cpu_throughput = (len(comp) * iterations / 1024**2) / cpu_elapsed
        print(f"    {cpu_throughput:.0f} MB/s (compressed)  =  {cpu_throughput/ratio:.0f} MB/s (uncompressed)")
        results["cpu_throughput_mbs_compressed"] = cpu_throughput
        results["cpu_throughput_mbs_uncompressed"] = cpu_throughput / ratio
    elif algorithm == "zstd":
        import zstandard as zstd
        print(f"\n  CPU decompression ({iterations} iterations):")
        t0 = time.perf_counter()
        for _ in range(iterations):
            zstd.ZstdDecompressor().decompress(comp)
        cpu_elapsed = time.perf_counter() - t0
        cpu_throughput = (len(comp) * iterations / 1024**2) / cpu_elapsed
        print(f"    {cpu_throughput:.0f} MB/s (compressed)  =  {cpu_throughput/ratio:.0f} MB/s (uncompressed)")
        results["cpu_throughput_mbs_compressed"] = cpu_throughput
        results["cpu_throughput_mbs_uncompressed"] = cpu_throughput / ratio
    elif algorithm == "lz4":
        try:
            import importlib.util
            _cw_path = Path(__file__).parent.parent / "tools" / "compress_weights.py"
            _spec = importlib.util.spec_from_file_location("compress_weights", _cw_path)
            _cw = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_cw)
            print(f"\n  CPU decompression ({iterations} iterations):")
            t0 = time.perf_counter()
            for _ in range(iterations):
                _cw._decompress_lz4(comp, len(raw))
            cpu_elapsed = time.perf_counter() - t0
            cpu_throughput = (len(comp) * iterations / 1024**2) / cpu_elapsed
            print(f"    {cpu_throughput:.0f} MB/s (compressed)  =  {cpu_throughput/ratio:.0f} MB/s (uncompressed)")
            results["cpu_throughput_mbs_compressed"] = cpu_throughput
            results["cpu_throughput_mbs_uncompressed"] = cpu_throughput / ratio
        except (ImportError, Exception) as e:
            print(f"\n  CPU decompression: skipped ({e})")
    else:
        # gdeflate has no CPU decompressor
        print(f"\n  CPU decompression: skipped ({algorithm} is GPU-only)")

    # GPU benchmark (if available)
    if torch.cuda.is_available():
        try:
            import vllm._C  # noqa: F401 — registers ops
            decompress_op = torch.ops._C.decompress_tensor
            has_nvcomp = bool(torch.ops._C.is_gpu_decompress_available())
            has_op = True
        except (ImportError, AttributeError):
            has_op = False
            has_nvcomp = False

        if has_op:
            mode = "nvCOMP GPU" if has_nvcomp else "GPU (CPU zlib fallback)"
            print(f"\n  GPU decompression via {mode} ({iterations} iterations):")
            comp_tensor = torch.frombuffer(
                bytearray(comp), dtype=torch.uint8
            ).cuda()
            dtype_str = str(tensor.dtype).replace("torch.", "")
            original_size = len(raw)
            shape = list(tensor.shape)

            # Warmup
            for _ in range(5):
                decompress_op(comp_tensor, shape, dtype_str, original_size,
                              algorithm)
            torch.cuda.synchronize()

            t0 = time.perf_counter()
            for _ in range(iterations):
                decompress_op(comp_tensor, shape, dtype_str, original_size,
                              algorithm)
            torch.cuda.synchronize()
            gpu_elapsed = time.perf_counter() - t0

            gpu_throughput = (len(comp) * iterations / 1024**2) / gpu_elapsed
            print(
                f"    {gpu_throughput:.0f} MB/s (compressed)  =  "
                f"{gpu_throughput/ratio:.0f} MB/s (uncompressed)"
            )
            if cpu_throughput:
                speedup = gpu_throughput / cpu_throughput
                print(f"    GPU speedup vs CPU: {speedup:.1f}x")
                results["gpu_speedup_vs_cpu"] = speedup
            results["gpu_throughput_mbs_compressed"] = gpu_throughput
            results["gpu_throughput_mbs_uncompressed"] = gpu_throughput / ratio
            results["nvcomp_available"] = has_nvcomp
        else:
            print("\n  GPU decompress op not available (build without nvCOMP support).")
    else:
        print("\n  No CUDA/GPU device available for GPU microbenchmark.")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark model weight compression for vLLM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to the original (uncompressed) model directory.",
    )
    parser.add_argument(
        "--compressed-model",
        type=Path,
        default=None,
        help="Path to pre-compressed model. If not given, compresses to a temp dir.",
    )
    parser.add_argument(
        "--algorithm",
        choices=["deflate", "zstd", "lz4", "gdeflate"],
        default="deflate",
        help="Compression algorithm to benchmark. "
             "'deflate' and 'zstd' use CPU. "
             "'lz4' and 'gdeflate' use GPU via nvCOMP (requires -DVLLM_NVCOMP_PATH=...). "
             "Default: deflate.",
    )
    parser.add_argument(
        "--level",
        type=int,
        default=6,
        help="Compression level. For deflate: 1-9. For zstd: 1-22. "
             "For gdeflate: 0=fast, 1=high-ratio, 2=entropy-only (values >2 are clamped to 2). "
             "Ignored for lz4. Default: 6.",
    )
    parser.add_argument(
        "--sections",
        nargs="+",
        choices=["compression", "loading", "memory", "inference", "decompress-microbench", "all"],
        default=["all"],
        help="Which benchmark sections to run. Default: all.",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["baseline", "compressed-load-all"],
        choices=["baseline", "compressed-load-all", "jit-cpu", "jit-gpu"],
        help="Inference backends to compare in Section 4.",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=20,
        help="Number of inference measurement steps. Default: 20.",
    )
    parser.add_argument(
        "--num-warmup",
        type=int,
        default=5,
        help="Number of warmup inference steps. Default: 5.",
    )
    parser.add_argument(
        "--cpu-offload-gb",
        type=float,
        default=0.0,
        help="GB of CPU RAM to use for offloading (Sections 3). Default: 0.",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        help="Model dtype for loading. Default: auto.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Write JSON results to this file.",
    )
    args = parser.parse_args()

    if not args.model.exists():
        parser.error(f"Model path does not exist: {args.model}")

    sections = set(args.sections)
    if "all" in sections:
        sections = {"compression", "loading", "memory", "inference", "decompress-microbench"}

    all_results: dict[str, Any] = {
        "model": str(args.model),
        "algorithm": args.algorithm,
        "level": args.level,
    }

    # Prepare compressed model (create temp if not provided)
    temp_dir = None
    compressed_path = args.compressed_model
    if compressed_path is None and (
        sections & {"loading", "memory", "inference"}
    ):
        temp_dir = tempfile.mkdtemp(prefix="vllm_compressed_bench_")
        compressed_path = Path(temp_dir) / "model_compressed"
        print(f"\nCompressing model to temporary directory: {compressed_path}")
        _make_compressed_model(args.model, compressed_path, args.algorithm, args.level)

    try:
        if "compression" in sections:
            all_results["compression"] = bench_compression_ratio(
                args.model, args.algorithm, args.level
            )

        if "loading" in sections and compressed_path:
            all_results["loading"] = bench_loading_time(
                args.model, compressed_path, args.dtype
            )

        if "memory" in sections and compressed_path:
            all_results["memory"] = bench_memory_footprint(
                args.model, compressed_path, args.cpu_offload_gb, args.dtype
            )

        if "inference" in sections and compressed_path:
            all_results["inference"] = bench_inference_throughput(
                args.model,
                compressed_path,
                backends=args.backends,
                num_warmup=args.num_warmup,
                num_steps=args.num_inference_steps,
                dtype=args.dtype,
            )

        if "decompress-microbench" in sections:
            all_results["decompress_microbench"] = bench_decompress_microbench(
                args.model, args.algorithm, args.level
            )

    finally:
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nResults saved to {args.output_json}")

    print("\nDone.")


if __name__ == "__main__":
    main()

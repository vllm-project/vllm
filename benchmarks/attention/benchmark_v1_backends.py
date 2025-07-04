#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmarking script for v1 attention backends under a variety of workloads.

This script benchmarks different attention backends
    (FlashAttention, FlashInfer, etc.)
across various batch configurations to measure performance characteristics.

Example usage:
    python benchmarks/attention/benchmark_v1_backends.py \
        --backends flash --specs q2k 8s1k 2q1k_32s1k
    python benchmarks/attention/benchmark_v1_backends.py \
        --backends flash --list-specs
"""

import argparse
import logging
import statistics
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Optional

import regex as re
import torch
from rich.console import Console
from rich.progress import Progress
from rich.table import Table

from vllm.config import (
    CacheConfig,
    CompilationConfig,
    DeviceConfig,
    LoadConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadataBuilder
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import FullAttentionSpec

# Optional imports for backends that may not be available
try:
    from vllm.v1.attention.backends.flashinfer import FlashInferMetadataBuilder

    FLASHINFER_AVAILABLE = True
except ImportError:
    FLASHINFER_AVAILABLE = False
    FlashInferMetadataBuilder = None

try:
    from vllm.v1.attention.backends.flex_attention import FlexAttentionMetadataBuilder

    FLEXATTENTION_AVAILABLE = True
except ImportError:
    FLEXATTENTION_AVAILABLE = False
    FlexAttentionMetadataBuilder = None

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_batch_spec(spec: str) -> list[tuple[int, int]]:
    """
    Grammar per segment (underscore separated):
      (<count>?) q<q_len>(k?) (s<kv_len>(k?))? : prefill/extend
      (<count>?) s<kv_len>(k?)            : decode
    'k' suffix multiplies by 1024.
    Examples:
      q2k -> [(2048,2048)]
      q2  -> [(2,2)]
      8s1k-> [(1,1024)]*8
      2q1k_32s1k -> [(1024,1024)]*2 + [(1,1024)]*32
    """
    pairs = []
    for seg in spec.split("_"):
        m = re.match(r"^(?:(\d+))?q(\d+)(k?)(?:s(\d+)(k?))?$", seg)
        if m:
            cnt = int(m.group(1)) if m.group(1) else 1
            q_len = int(m.group(2))
            qlen = q_len * 1024 if m.group(3) == "k" else q_len
            if m.group(4):
                kv_len = int(m.group(4))
                klen = kv_len * 1024 if m.group(5) == "k" else kv_len
            else:
                klen = qlen
            pairs.extend([(qlen, klen)] * cnt)
            continue
        m = re.match(r"^(?:(\d+))?s(\d+)(k?)$", seg)
        if m:
            cnt = int(m.group(1)) if m.group(1) else 1
            kv_len = int(m.group(2))
            klen = kv_len * 1024 if m.group(3) == "k" else kv_len
            pairs.extend([(1, klen)] * cnt)
            continue
        raise argparse.ArgumentTypeError(f"Invalid batch spec '{seg}'")
    return pairs


def format_batch_spec(pairs: list[tuple[int, int]]) -> str:
    """Pretty-print list[(q,kv)] into human-readable segments."""
    kinds: dict[str, list[tuple[int, int]]] = {
        "prefill": [],
        "extend": [],
        "specdecode": [],
        "decode": [],
        "unknown": [],
    }
    for q, kv in pairs:
        if q > 1 and kv == q:
            kinds["prefill"].append((q, kv))
        elif q > 1 and kv > q:
            kinds["extend"].append((q, kv))
        elif q > 1 and q <= 16:
            kinds["specdecode"].append((q, kv))
        elif q == 1 and kv > 1:
            kinds["decode"].append((q, kv))
        else:
            kinds["unknown"].append((q, kv))
    parts = []
    for kind in ["prefill", "extend", "specdecode", "decode", "unknown"]:
        lst = kinds[kind]
        if not lst:
            continue
        cnt_total = len(lst)
        ctr = Counter(lst)
        inner = []
        for (q, kv), cnt in ctr.items():
            if kind == "prefill":
                size = f"{q // 1024}k" if q % 1024 == 0 else str(q)
                inner.append(f"{cnt}x{size}")
            elif kind == "decode":
                size = f"{kv // 1024}k" if kv % 1024 == 0 else str(kv)
                inner.append(f"{cnt}x{size}")
            else:
                qstr = f"{q // 1024}k" if q % 1024 == 0 else str(q)
                kstr = f"{kv // 1024}k" if kv % 1024 == 0 else str(kv)
                inner.append(f"{cnt}xq{qstr}s{kstr}")
        parts.append(f"{cnt_total} {kind} ({', '.join(inner)})")
    return ", ".join(parts)


@dataclass
class BatchSpec:
    """Specification for a batch configuration."""

    name: str
    description: str
    batch_size: int
    num_tokens: int
    seq_lens: list[int]
    query_lens: list[int]
    block_size: int = 16
    num_kv_heads: int = 8
    head_size: int = 64
    dtype: torch.dtype = torch.float16
    use_mla: bool = False
    sliding_window: Optional[int] = None

    def __post_init__(self):
        assert len(self.seq_lens) == self.batch_size
        assert len(self.query_lens) == self.batch_size
        assert sum(self.query_lens) == self.num_tokens

    @classmethod
    def from_spec_string(cls, spec_str: str, **kwargs) -> "BatchSpec":
        """Create BatchSpec from a spec string like 'q2k' or '8s1k'."""
        pairs = parse_batch_spec(spec_str)
        description = format_batch_spec(pairs)

        batch_size = len(pairs)
        query_lens = [q for q, _ in pairs]
        seq_lens = [kv for _, kv in pairs]
        num_tokens = sum(query_lens)

        return cls(
            name=spec_str,
            description=description,
            batch_size=batch_size,
            num_tokens=num_tokens,
            seq_lens=seq_lens,
            query_lens=query_lens,
            **kwargs,
        )


# Define some common benchmark specs for easy reference
DEFAULT_BENCHMARK_SPECS = [
    "q2k",  # 1 prefill (1x2k)
    "8s1k",  # 8 decode (8x1k)
    "q1k",  # 1 prefill (1x1k)
    "16s2k",  # 16 decode (16x2k)
    "2q1k_32s1k",  # 2 prefill (2x1k), 32 decode (32x1k)
    "32q4s1k",  # 32 extend (32xq4s1k)
    "4s32k",  # 4 decode (4x32k)
    "64s2k",  # 64 decode (64x2k)
    "16q1k",  # 16 prefill (16x1k)
    "8q2k",  # 8 prefill (8x2k)
]


class AttentionBenchmarker:
    """Benchmarks attention backends with different configurations."""

    def __init__(
        self, device: torch.device, warmup_runs: int = 3, benchmark_runs: int = 10
    ):
        self.device = device
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs
        self.console = Console()

        # Create base VllmConfig
        self.base_vllm_config = self._create_vllm_config()

        # Available backends
        self.backends: dict[str, tuple[str, Any]] = {
            "flash": ("FlashAttention", FlashAttentionMetadataBuilder),
        }

        # Note: FlashInfer and FlexAttention may not be refactored yet
        if FLASHINFER_AVAILABLE:
            self.backends["flashinfer"] = ("FlashInfer", FlashInferMetadataBuilder)

        if FLEXATTENTION_AVAILABLE:
            self.backends["flex"] = ("FlexAttention", FlexAttentionMetadataBuilder)

    def _create_vllm_config(self) -> VllmConfig:
        """Create a base VllmConfig for benchmarking."""
        model_config = ModelConfig(
            model="facebook/opt-125m",
            max_model_len=2048,  # Use the model's actual max length
            dtype=torch.float16,
        )
        cache_config = CacheConfig(
            block_size=16,
            cache_dtype="auto",
        )
        parallel_config = ParallelConfig()
        scheduler_config = SchedulerConfig(
            max_num_seqs=128,
            max_num_batched_tokens=32768,
        )
        device_config = DeviceConfig()
        load_config = LoadConfig()
        compilation_config = CompilationConfig()

        return VllmConfig(
            model_config=model_config,
            cache_config=cache_config,
            parallel_config=parallel_config,
            scheduler_config=scheduler_config,
            device_config=device_config,
            load_config=load_config,
            compilation_config=compilation_config,
        )

    def _create_kv_cache_spec(self, batch_spec: BatchSpec) -> FullAttentionSpec:
        """Create KV cache specification for the batch."""
        return FullAttentionSpec(
            block_size=batch_spec.block_size,
            num_kv_heads=batch_spec.num_kv_heads,
            head_size=batch_spec.head_size,
            dtype=batch_spec.dtype,
            use_mla=batch_spec.use_mla,
            sliding_window=batch_spec.sliding_window,
        )

    def _create_common_attn_metadata(
        self, batch_spec: BatchSpec
    ) -> CommonAttentionMetadata:
        """Create CommonAttentionMetadata for the batch specification."""
        # Calculate blocks needed for each sequence
        blocks_per_seq = []
        for seq_len in batch_spec.seq_lens:
            blocks_needed = (
                seq_len + batch_spec.block_size - 1
            ) // batch_spec.block_size
            blocks_per_seq.append(blocks_needed)

        # Create block tables (simplified - just sequential block IDs)
        max_blocks = max(blocks_per_seq)
        block_table_tensor = torch.zeros(
            (batch_spec.batch_size, max_blocks), dtype=torch.int32, device=self.device
        )
        current_block = 0
        for i, blocks_needed in enumerate(blocks_per_seq):
            for j in range(blocks_needed):
                block_table_tensor[i, j] = current_block + j
            current_block += blocks_needed

        # Create slot mapping (token -> block_id * block_size + offset)
        slot_mapping = []
        for i, (seq_len, query_len) in enumerate(
            zip(batch_spec.seq_lens, batch_spec.query_lens)
        ):
            start_block = sum(blocks_per_seq[:i])
            for token_idx in range(query_len):
                pos_in_seq = seq_len - query_len + token_idx
                block_id = start_block + pos_in_seq // batch_spec.block_size
                offset = pos_in_seq % batch_spec.block_size
                slot_mapping.append(block_id * batch_spec.block_size + offset)

        # Create query start locations
        query_start_loc = torch.zeros(
            batch_spec.batch_size + 1, dtype=torch.int32, device=self.device
        )
        query_start_loc[1:] = torch.tensor(
            batch_spec.query_lens, dtype=torch.int32, device=self.device
        ).cumsum(0)
        query_start_loc_cpu = query_start_loc.cpu()

        # Create sequence lengths
        seq_lens = torch.tensor(
            batch_spec.seq_lens, dtype=torch.int32, device=self.device
        )
        seq_lens_cpu = seq_lens.cpu()

        # Create computed tokens (assume context tokens are computed)
        num_computed_tokens_cpu = torch.tensor(
            [
                seq_len - query_len
                for seq_len, query_len in zip(
                    batch_spec.seq_lens, batch_spec.query_lens
                )
            ],
            dtype=torch.int32,
        )

        # Create slot mapping tensors
        slot_mapping_tensor = torch.tensor(
            slot_mapping, dtype=torch.long, device=self.device
        )

        return CommonAttentionMetadata(
            query_start_loc=query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            num_computed_tokens_cpu=num_computed_tokens_cpu,
            num_reqs=batch_spec.batch_size,
            num_actual_tokens=batch_spec.num_tokens,
            max_query_len=max(batch_spec.query_lens),
            block_table_tensor=block_table_tensor,
            slot_mapping=slot_mapping_tensor,
        )

    def _benchmark_backend(self, backend_name: str, batch_spec: BatchSpec) -> float:
        """Benchmark a specific backend with a batch specification."""
        if backend_name not in self.backends:
            raise ValueError(f"Unknown backend: {backend_name}")

        _, metadata_builder_cls = self.backends[backend_name]

        # Create KV cache spec and common metadata
        kv_cache_spec = self._create_kv_cache_spec(batch_spec)
        common_metadata = self._create_common_attn_metadata(batch_spec)

        # Create the metadata builder
        metadata_builder = metadata_builder_cls(
            kv_cache_spec=kv_cache_spec,
            vllm_config=self.base_vllm_config,
            device=self.device,
        )

        # Build attention metadata
        attn_metadata = metadata_builder.build(
            common_prefix_len=0,
            common_attn_metadata=common_metadata,
        )

        # Create dummy query, key, value tensors
        total_tokens = batch_spec.num_tokens
        num_heads = batch_spec.num_kv_heads * 4  # Assume 4:1 query:kv head ratio

        # For FlashAttention, query, key, value must have the same batch dimension
        # We only pass the new tokens being processed
        query = torch.randn(
            total_tokens,
            num_heads,
            batch_spec.head_size,
            dtype=batch_spec.dtype,
            device=self.device,
        )
        key = torch.randn(
            total_tokens,
            batch_spec.num_kv_heads,
            batch_spec.head_size,
            dtype=batch_spec.dtype,
            device=self.device,
        )
        value = torch.randn(
            total_tokens,
            batch_spec.num_kv_heads,
            batch_spec.head_size,
            dtype=batch_spec.dtype,
            device=self.device,
        )

        # Create dummy KV cache
        total_blocks = sum(
            (seq_len + batch_spec.block_size - 1) // batch_spec.block_size
            for seq_len in batch_spec.seq_lens
        )
        kv_cache = torch.randn(
            2,
            total_blocks,
            batch_spec.block_size,
            batch_spec.num_kv_heads,
            batch_spec.head_size,
            dtype=batch_spec.dtype,
            device=self.device,
        )

        # Create the backend implementation (FlashAttention impl)
        from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl

        backend = FlashAttentionImpl(
            num_heads=num_heads,
            head_size=batch_spec.head_size,
            scale=1.0,  # Default scale
            num_kv_heads=batch_spec.num_kv_heads,
            alibi_slopes=None,
            sliding_window=batch_spec.sliding_window,
            kv_cache_dtype="auto",
            logits_soft_cap=None,
        )

        # Create a dummy layer with q_scale, k_scale and v_scale attributes
        class DummyLayer(torch.nn.Module):
            def __init__(self, device):
                super().__init__()
                self._q_scale = torch.tensor(1.0, device=device)
                self._k_scale = torch.tensor(1.0, device=device)
                self._v_scale = torch.tensor(1.0, device=device)

        dummy_layer = DummyLayer(self.device)

        # Warmup runs
        for _ in range(self.warmup_runs):
            try:
                output = torch.empty(
                    total_tokens,
                    num_heads,
                    batch_spec.head_size,
                    dtype=batch_spec.dtype,
                    device=self.device,
                )
                _ = backend.forward(
                    layer=dummy_layer,
                    query=query,
                    key=key,
                    value=value,
                    kv_cache=kv_cache,
                    attn_metadata=attn_metadata,
                    output=output,
                )
                torch.cuda.synchronize()
            except Exception as e:
                logger.warning(
                    "Warmup failed for %s with %s: %s",
                    backend_name,
                    batch_spec.name,
                    e,
                )
                return float("inf")

        # Benchmark runs
        times = []
        for _ in range(self.benchmark_runs):
            torch.cuda.synchronize()
            start_time = time.perf_counter()

            try:
                output = torch.empty(
                    total_tokens,
                    num_heads,
                    batch_spec.head_size,
                    dtype=batch_spec.dtype,
                    device=self.device,
                )
                _ = backend.forward(
                    layer=dummy_layer,
                    query=query,
                    key=key,
                    value=value,
                    kv_cache=kv_cache,
                    attn_metadata=attn_metadata,
                    output=output,
                )
                torch.cuda.synchronize()
                end_time = time.perf_counter()
                times.append(end_time - start_time)
            except Exception as e:
                logger.warning(
                    "Benchmark failed for %s with %s: %s",
                    backend_name,
                    batch_spec.name,
                    e,
                )
                return float("inf")

        # Return median time
        return statistics.median(times)

    def benchmark(
        self, backend_names: list[str], spec_strings: list[str]
    ) -> dict[str, dict[str, float]]:
        """Run benchmarks for specified backends and batch specifications."""
        # Convert spec strings to BatchSpec objects
        batch_specs = []
        for spec_str in spec_strings:
            try:
                batch_spec = BatchSpec.from_spec_string(spec_str)
                batch_specs.append(batch_spec)
            except argparse.ArgumentTypeError as e:
                logger.error("Invalid batch spec '%s': %s", spec_str, e)
                continue

        if not batch_specs:
            raise ValueError("No valid batch specifications provided")

        results = {}

        with Progress() as progress:
            total_tasks = len(backend_names) * len(batch_specs)
            task = progress.add_task("Benchmarking...", total=total_tasks)

            for backend_name in backend_names:
                if backend_name not in self.backends:
                    logger.warning("Unknown backend: %s, skipping", backend_name)
                    progress.advance(task, len(batch_specs))
                    continue

                results[backend_name] = {}

                for batch_spec in batch_specs:
                    logger.info(
                        "Benchmarking %s with %s (%s)",
                        backend_name,
                        batch_spec.name,
                        batch_spec.description,
                    )

                    try:
                        time_taken = self._benchmark_backend(backend_name, batch_spec)
                        results[backend_name][batch_spec.name] = time_taken
                        logger.info("  Result: %.6fs", time_taken)
                    except Exception as e:
                        logger.error("  Failed: %s", e)
                        results[backend_name][batch_spec.name] = float("inf")

                    progress.advance(task, 1)

        return results

    def print_results(
        self,
        results: dict[str, dict[str, float]],
        backend_names: list[str],
        spec_strings: list[str],
    ):
        """Print benchmark results in a formatted table."""
        # Convert spec strings to descriptions
        spec_descriptions = {}
        for spec_str in spec_strings:
            try:
                pairs = parse_batch_spec(spec_str)
                description = format_batch_spec(pairs)
                spec_descriptions[spec_str] = description
            except argparse.ArgumentTypeError:
                spec_descriptions[spec_str] = spec_str

        table = Table(title="Attention Benchmark")
        table.add_column("BatchSpec", style="cyan", no_wrap=True)

        # Add columns for each backend
        for backend_name in backend_names:
            if backend_name in results:
                table.add_column(f"{backend_name} Time (s)", style="green")

        # Add relative performance columns
        if len([b for b in backend_names if b in results]) > 1:
            for backend_name in backend_names:
                if backend_name in results:
                    table.add_column(f"{backend_name} % of Fastest", style="yellow")

        # Add rows
        for spec_str in spec_strings:
            if not any(spec_str in results.get(b, {}) for b in backend_names):
                continue

            row = [f"{spec_str}\n({spec_descriptions[spec_str]})"]

            # Get times for this spec across all backends
            spec_times = {}
            for backend_name in backend_names:
                if backend_name in results and spec_str in results[backend_name]:
                    time_val = results[backend_name][spec_str]
                    spec_times[backend_name] = (
                        time_val if time_val != float("inf") else None
                    )

            # Add time columns
            for backend_name in backend_names:
                if backend_name in results:
                    time_val = spec_times.get(backend_name)
                    if time_val is not None:
                        row.append(f"{time_val:.6f}")
                    else:
                        row.append("FAILED")

            # Add relative performance columns
            if len([b for b in backend_names if b in results]) > 1:
                valid_times = [t for t in spec_times.values() if t is not None]
                if valid_times:
                    fastest_time = min(valid_times)
                    for backend_name in backend_names:
                        if backend_name in results:
                            time_val = spec_times.get(backend_name)
                            if time_val is not None:
                                percentage = (time_val / fastest_time) * 100
                                row.append(f"{percentage:.1f}%")
                            else:
                                row.append("N/A")

            table.add_row(*row)

        self.console.print(table)


def main():
    parser = argparse.ArgumentParser(description="Benchmark v1 attention backends")
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["flash"],
        choices=["flash", "flashinfer", "flex"],
        help="Attention backends to benchmark",
    )
    parser.add_argument(
        "--specs",
        nargs="+",
        default=DEFAULT_BENCHMARK_SPECS[:5],  # Use first 5 default specs
        help="Batch specifications to benchmark (e.g., 'q2k', '8s1k', '2q1k_32s1k')",
    )
    parser.add_argument(
        "--list-specs",
        action="store_true",
        help="List all default batch specifications and exit",
    )
    parser.add_argument(
        "--warmup-runs", type=int, default=3, help="Number of warmup runs per benchmark"
    )
    parser.add_argument(
        "--benchmark-runs",
        type=int,
        default=10,
        help="Number of benchmark runs per test",
    )
    parser.add_argument("--device", default="cuda", help="Device to run benchmarks on")

    args = parser.parse_args()

    if args.list_specs:
        print("Default batch specifications:")
        for spec in DEFAULT_BENCHMARK_SPECS:
            try:
                pairs = parse_batch_spec(spec)
                description = format_batch_spec(pairs)
                print(f"  {spec:15} -> {description}")
            except Exception as e:
                print(f"  {spec:15} -> ERROR: {e}")
        return

    # Check device availability
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    # Create benchmarker
    benchmarker = AttentionBenchmarker(
        device=device, warmup_runs=args.warmup_runs, benchmark_runs=args.benchmark_runs
    )

    # Run benchmarks
    logger.info("Running benchmarks on %s", device)
    logger.info("Backends: %s", args.backends)
    logger.info("Specs: %s", args.specs)

    results = benchmarker.benchmark(args.backends, args.specs)

    # Print results
    benchmarker.print_results(results, args.backends, args.specs)


if __name__ == "__main__":
    main()

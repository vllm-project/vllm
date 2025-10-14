# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Common utilities for attention benchmarking."""

import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from rich.console import Console
from rich.table import Table

# Mock classes for vLLM attention infrastructure


class MockHfConfig:
    """Mock HuggingFace config that satisfies vLLM's requirements."""

    def __init__(self, mla_dims: dict):
        self.num_attention_heads = mla_dims["num_q_heads"]
        self.num_key_value_heads = mla_dims["num_kv_heads"]
        self.hidden_size = mla_dims["head_dim"] * mla_dims["num_q_heads"]
        self.model_type = "deepseek_v2"
        self.is_encoder_decoder = False

    def get_text_config(self):
        return self


# Import AttentionLayerBase at module level to avoid circular dependencies
try:
    from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase

    _HAS_ATTENTION_LAYER_BASE = True
except ImportError:
    _HAS_ATTENTION_LAYER_BASE = False
    AttentionLayerBase = object  # Fallback


class MockLayer(AttentionLayerBase):
    """Mock attention layer with scale parameters and impl.

    Inherits from AttentionLayerBase so it passes isinstance checks
    in get_layers_from_vllm_config when FlashInfer prefill is enabled.
    """

    def __init__(self, device: torch.device, impl=None):
        # Don't call super().__init__() as AttentionLayerBase doesn't have __init__
        self._k_scale = torch.tensor(1.0, device=device)
        self._v_scale = torch.tensor(1.0, device=device)
        self._q_scale = torch.tensor(1.0, device=device)
        # Scalar floats for kernels that need them
        self._k_scale_float = float(self._k_scale.item())
        self._v_scale_float = float(self._v_scale.item())
        self._q_scale_float = float(self._q_scale.item())
        # AttentionImpl for metadata builders to query
        self.impl = impl

    def get_attn_backend(self):
        """Get the attention backend class (required by AttentionLayerBase)."""
        # Return None as this is just a mock layer for benchmarking
        return None


class MockModelConfig:
    """Mock model configuration."""

    def __init__(
        self,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16,
        max_model_len: int = 32768,
    ):
        self._n_q = num_q_heads
        self._n_kv = num_kv_heads
        self._d = head_dim
        self.dtype = dtype
        self.max_model_len = max_model_len

    def get_num_attention_heads(self, _=None) -> int:
        return self._n_q

    def get_num_kv_heads(self, _=None) -> int:
        return self._n_kv

    def get_head_size(self) -> int:
        return self._d

    def get_num_layers(self) -> int:
        """Mock method for layer count queries."""
        return 1

    def get_sliding_window_for_layer(self, _layer_idx: int):
        """Mock method for sliding window queries."""
        return None

    def get_logits_soft_cap_for_layer(self, _layer_idx: int):
        """Mock method for logits soft cap queries."""
        return None

    def get_sm_scale_for_layer(self, _layer_idx: int) -> float:
        """Mock method for SM scale queries."""
        return 1.0 / (self.get_head_size() ** 0.5)


class MockParallelConfig:
    """Mock parallel configuration."""

    pass


class MockCompilationConfig:
    """Mock compilation configuration."""

    def __init__(self):
        self.full_cuda_graph = False
        self.static_forward_context = {}


class MockVLLMConfig:
    """Mock VLLM configuration."""

    def __init__(self):
        self.compilation_config = MockCompilationConfig()


class MockRunner:
    """Mock GPU runner for metadata builders."""

    def __init__(
        self,
        seq_lens: np.ndarray,
        query_start_locs: np.ndarray,
        device: torch.device,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
    ):
        self.model_config = MockModelConfig(num_q_heads, num_kv_heads, head_dim, dtype)
        self.parallel_config = MockParallelConfig()
        self.vllm_config = MockVLLMConfig()
        self.seq_lens_np = seq_lens
        self.query_start_loc_np = query_start_locs
        self.device = device
        self.attention_chunk_size = None
        self.num_query_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.dtype = dtype


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark run."""

    backend: str
    batch_spec: str
    num_layers: int
    head_dim: int
    num_q_heads: int
    num_kv_heads: int
    block_size: int
    device: str
    dtype: torch.dtype = torch.float16
    repeats: int = 1
    warmup_iters: int = 3
    profile_memory: bool = False
    use_cuda_graphs: bool = False

    # MLA-specific
    kv_lora_rank: Optional[int] = None
    qk_nope_head_dim: Optional[int] = None
    qk_rope_head_dim: Optional[int] = None
    v_head_dim: Optional[int] = None

    # Backend-specific tuning
    num_kv_splits: Optional[int] = None  # CUTLASS MLA
    reorder_batch_threshold: Optional[int] = None  # FlashAttn MLA, FlashMLA


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    config: BenchmarkConfig
    mean_time: float  # seconds
    std_time: float  # seconds
    min_time: float  # seconds
    max_time: float  # seconds
    throughput_tokens_per_sec: Optional[float] = None
    memory_allocated_mb: Optional[float] = None
    memory_reserved_mb: Optional[float] = None
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        """Whether benchmark completed successfully."""
        return self.error is None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "config": asdict(self.config),
            "mean_time": self.mean_time,
            "std_time": self.std_time,
            "min_time": self.min_time,
            "max_time": self.max_time,
            "throughput_tokens_per_sec": self.throughput_tokens_per_sec,
            "memory_allocated_mb": self.memory_allocated_mb,
            "memory_reserved_mb": self.memory_reserved_mb,
            "error": self.error,
        }


class BenchmarkRunner:
    """Base class for running attention benchmarks."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.device = torch.device(config.device)
        torch.cuda.set_device(self.device)

    def run(self, **kwargs) -> BenchmarkResult:
        """
        Run benchmark with current configuration.

        Returns:
            BenchmarkResult with timing and memory statistics
        """
        raise NotImplementedError

    def _time_kernel(self, fn, warmup: int = 3, repeats: int = 10) -> dict:
        """
        Time a kernel function with warmup and multiple repeats.

        Args:
            fn: Callable to time
            warmup: Number of warmup iterations
            repeats: Number of measurement iterations

        Returns:
            Dict with timing statistics
        """
        # Warmup
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()

        # Measure
        times = []
        for _ in range(repeats):
            torch.cuda.synchronize()
            start = time.time()
            fn()
            torch.cuda.synchronize()
            times.append(time.time() - start)

        return {
            "mean": np.mean(times),
            "std": np.std(times),
            "min": np.min(times),
            "max": np.max(times),
        }

    def _get_memory_stats(self) -> dict:
        """Get current CUDA memory statistics."""
        return {
            "allocated_mb": torch.cuda.memory_allocated(self.device) / 1024**2,
            "reserved_mb": torch.cuda.memory_reserved(self.device) / 1024**2,
        }


class ResultsFormatter:
    """Format and display benchmark results."""

    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()

    def print_table(
        self,
        results: list[BenchmarkResult],
        backends: list[str],
        compare_to_fastest: bool = True,
    ):
        """
        Print results as a rich table.

        Args:
            results: List of BenchmarkResult
            backends: List of backend names being compared
            compare_to_fastest: Show percentage comparison to fastest
        """
        # Group by batch spec
        by_spec = {}
        for r in results:
            spec = r.config.batch_spec
            if spec not in by_spec:
                by_spec[spec] = {}
            by_spec[spec][r.config.backend] = r

        table = Table(title="Attention Benchmark Results")
        table.add_column("Batch Spec", no_wrap=True)

        multi = len(backends) > 1
        for backend in backends:
            # Time column
            col_time = f"{backend} Time (s)"
            table.add_column(col_time, justify="right", no_wrap=True)
            if multi and compare_to_fastest:
                # Relative performance column
                col_rel = f"{backend} vs Fastest"
                table.add_column(col_rel, justify="right", no_wrap=True)

        # Add rows
        for spec in sorted(by_spec.keys()):
            spec_results = by_spec[spec]
            times = {b: r.mean_time for b, r in spec_results.items() if r.success}
            best_time = min(times.values()) if times else 0.0

            row = [spec]
            for backend in backends:
                if backend in spec_results:
                    r = spec_results[backend]
                    if r.success:
                        row.append(f"{r.mean_time:.6f}")
                        if multi and compare_to_fastest:
                            pct = (
                                (r.mean_time / best_time * 100) if best_time > 0 else 0
                            )
                            pct_str = f"{pct:.1f}%"
                            if r.mean_time == best_time:
                                pct_str = f"[bold green]{pct_str}[/]"
                            row.append(pct_str)
                    else:
                        row.append("[red]ERROR[/]")
                        if multi and compare_to_fastest:
                            row.append("-")
                else:
                    row.append("-")
                    if multi and compare_to_fastest:
                        row.append("-")

            table.add_row(*row)

        self.console.print(table)

    def save_csv(self, results: list[BenchmarkResult], path: str):
        """Save results to CSV file."""
        if not results:
            return

        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "backend",
                    "batch_spec",
                    "num_layers",
                    "mean_time",
                    "std_time",
                    "throughput",
                    "memory_mb",
                ],
            )
            writer.writeheader()
            for r in results:
                writer.writerow(
                    {
                        "backend": r.config.backend,
                        "batch_spec": r.config.batch_spec,
                        "num_layers": r.config.num_layers,
                        "mean_time": r.mean_time,
                        "std_time": r.std_time,
                        "throughput": r.throughput_tokens_per_sec or 0,
                        "memory_mb": r.memory_allocated_mb or 0,
                    }
                )

        self.console.print(f"[green]Saved CSV results to {path}[/]")

    def save_json(self, results: list[BenchmarkResult], path: str):
        """Save results to JSON file."""
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        data = [r.to_dict() for r in results]
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        self.console.print(f"[green]Saved JSON results to {path}[/]")


def setup_mla_dims(model_name: str = "deepseek-v3") -> dict:
    """
    Get MLA dimensions for known models.

    Args:
        model_name: Model identifier

    Returns:
        Dict with MLA dimension configuration
    """
    configs = {
        "deepseek-v2": {
            "kv_lora_rank": 512,
            "qk_nope_head_dim": 128,
            "qk_rope_head_dim": 64,
            "v_head_dim": 128,
            "num_q_heads": 128,
            "num_kv_heads": 1,
            "head_dim": 576,
        },
        "deepseek-v3": {
            "kv_lora_rank": 512,
            "qk_nope_head_dim": 128,
            "qk_rope_head_dim": 64,
            "v_head_dim": 128,
            "num_q_heads": 128,
            "num_kv_heads": 1,
            "head_dim": 576,
        },
        "deepseek-v2-lite": {
            "kv_lora_rank": 512,
            "qk_nope_head_dim": 128,
            "qk_rope_head_dim": 64,
            "v_head_dim": 128,
            "num_q_heads": 16,
            "num_kv_heads": 1,
            "head_dim": 576,
        },
    }

    if model_name not in configs:
        raise ValueError(
            f"Unknown model '{model_name}'. Known models: {list(configs.keys())}"
        )

    return configs[model_name]


def get_attention_scale(head_dim: int) -> float:
    """Compute attention scale factor (1/sqrt(d))."""
    return 1.0 / math.sqrt(head_dim)

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Common utilities for attention benchmarking."""

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from batch_spec import get_batch_type, parse_batch_spec
from rich.console import Console
from rich.table import Table

# Mock classes for vLLM attention infrastructure


class MockHfConfig:
    """Mock HuggingFace config that satisfies vLLM's requirements."""

    def __init__(self, mla_dims: dict, index_topk: int | None = None):
        self.num_attention_heads = mla_dims["num_q_heads"]
        self.num_key_value_heads = mla_dims["num_kv_heads"]
        self.hidden_size = mla_dims["head_dim"] * mla_dims["num_q_heads"]
        self.model_type = "deepseek_v2"
        self.is_encoder_decoder = False
        self.kv_lora_rank = mla_dims["kv_lora_rank"]
        self.qk_nope_head_dim = mla_dims["qk_nope_head_dim"]
        self.qk_rope_head_dim = mla_dims["qk_rope_head_dim"]
        self.v_head_dim = mla_dims["v_head_dim"]
        self.qk_head_dim = mla_dims["qk_nope_head_dim"] + mla_dims["qk_rope_head_dim"]
        if index_topk is not None:
            self.index_topk = index_topk

    def get_text_config(self):
        return self


# Import AttentionLayerBase at module level to avoid circular dependencies
try:
    from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase

    _HAS_ATTENTION_LAYER_BASE = True
except ImportError:
    _HAS_ATTENTION_LAYER_BASE = False
    AttentionLayerBase = object  # Fallback


class MockKVBProj:
    """Mock KV projection layer for MLA prefill mode.

    Mimics ColumnParallelLinear behavior for kv_b_proj in MLA backends.
    Projects kv_c_normed to [qk_nope_head_dim + v_head_dim] per head.
    """

    def __init__(self, num_heads: int, qk_nope_head_dim: int, v_head_dim: int):
        self.num_heads = num_heads
        self.qk_nope_head_dim = qk_nope_head_dim
        self.v_head_dim = v_head_dim
        self.out_dim = qk_nope_head_dim + v_head_dim

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        """
        Project kv_c_normed to output space.

        Args:
            x: Input tensor [num_tokens, kv_lora_rank]

        Returns:
            Tuple containing output tensor
                [num_tokens, num_heads, qk_nope_head_dim + v_head_dim]
        """
        num_tokens = x.shape[0]
        result = torch.randn(
            num_tokens,
            self.num_heads,
            self.out_dim,
            device=x.device,
            dtype=x.dtype,
        )
        return (result,)  # Return as tuple to match ColumnParallelLinear API


class MockIndexer:
    """Mock Indexer for sparse MLA backends.

    Provides topk_indices_buffer that sparse MLA backends use to determine
    which KV cache slots to attend to for each token.
    """

    def __init__(
        self,
        max_num_tokens: int,
        topk_tokens: int,
        device: torch.device,
    ):
        self.topk_tokens = topk_tokens
        self.topk_indices_buffer = torch.zeros(
            (max_num_tokens, topk_tokens),
            dtype=torch.int32,
            device=device,
        )

    def fill_random_indices(self, num_tokens: int, max_kv_len: int):
        """Fill topk_indices_buffer with random valid indices for benchmarking."""
        indices = torch.randint(
            0,
            max_kv_len,
            (num_tokens, self.topk_tokens),
            dtype=torch.int32,
            device=self.topk_indices_buffer.device,
        )
        self.topk_indices_buffer[:num_tokens] = indices


class MockLayer(AttentionLayerBase):
    """Mock attention layer with scale parameters and impl.

    Inherits from AttentionLayerBase so it passes isinstance checks
    in get_layers_from_vllm_config when FlashInfer prefill is enabled.
    """

    def __init__(self, device: torch.device, impl=None, kv_cache_spec=None):
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
        # KV cache spec for get_kv_cache_spec
        self._kv_cache_spec = kv_cache_spec

    def get_attn_backend(self):
        """Get the attention backend class (required by AttentionLayerBase)."""
        # Return None as this is just a mock layer for benchmarking
        return None

    def get_kv_cache_spec(self):
        """Get the KV cache spec (required by AttentionLayerBase)."""
        return self._kv_cache_spec


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
class ParameterSweep:
    """Configuration for sweeping a backend parameter."""

    param_name: str  # Name of the backend parameter to sweep
    values: list[Any]  # List of values to test
    include_auto: bool = False  # Also test with param unset (auto mode)
    label_format: str = "{backend}_{param_name}_{value}"  # Result label template

    def get_label(self, backend: str, value: Any) -> str:
        """Generate a label for a specific parameter value."""
        return self.label_format.format(
            backend=backend, param_name=self.param_name, value=value
        )


@dataclass
class ModelParameterSweep:
    """Configuration for sweeping a model configuration parameter."""

    param_name: str  # Name of the model config parameter to sweep (e.g., "num_q_heads")
    values: list[Any]  # List of values to test
    label_format: str = "{backend}_{param_name}_{value}"  # Result label template

    def get_label(self, backend: str, value: Any) -> str:
        """Generate a label for a specific parameter value."""
        return self.label_format.format(
            backend=backend, param_name=self.param_name, value=value
        )


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
    kv_lora_rank: int | None = None
    qk_nope_head_dim: int | None = None
    qk_rope_head_dim: int | None = None
    v_head_dim: int | None = None

    # Backend-specific tuning
    num_kv_splits: int | None = None  # CUTLASS MLA
    reorder_batch_threshold: int | None = None  # FlashAttn MLA, FlashMLA


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    config: BenchmarkConfig
    mean_time: float  # seconds
    std_time: float  # seconds
    min_time: float  # seconds
    max_time: float  # seconds
    throughput_tokens_per_sec: float | None = None
    memory_allocated_mb: float | None = None
    memory_reserved_mb: float | None = None
    error: str | None = None

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


class ResultsFormatter:
    """Format and display benchmark results."""

    def __init__(self, console: Console | None = None):
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
        # Group by batch spec, preserving first-occurrence order
        by_spec = {}
        specs_order = []
        for r in results:
            spec = r.config.batch_spec
            if spec not in by_spec:
                by_spec[spec] = {}
                specs_order.append(spec)
            by_spec[spec][r.config.backend] = r

        # Create shortened backend names for display
        def shorten_backend_name(name: str) -> str:
            """Shorten long backend names for table display."""
            # Remove common prefixes
            name = name.replace("flashattn_mla", "famla")
            name = name.replace("flashinfer_mla", "fimla")
            name = name.replace("flashmla", "fmla")
            name = name.replace("cutlass_mla", "cmla")
            name = name.replace("numsplits", "ns")
            return name

        table = Table(title="Attention Benchmark Results")
        table.add_column("Batch\nSpec", no_wrap=True)
        table.add_column("Type", no_wrap=True)
        table.add_column("Batch\nSize", justify="right", no_wrap=True)

        multi = len(backends) > 1
        for backend in backends:
            short_name = shorten_backend_name(backend)
            # Time column
            col_time = f"{short_name}\nTime (s)"
            table.add_column(col_time, justify="right", no_wrap=False)
            if multi and compare_to_fastest:
                # Relative performance column
                col_rel = f"{short_name}\nvs Best"
                table.add_column(col_rel, justify="right", no_wrap=False)

        # Add rows
        for spec in specs_order:
            spec_results = by_spec[spec]
            times = {b: r.mean_time for b, r in spec_results.items() if r.success}
            best_time = min(times.values()) if times else 0.0

            batch_type = get_batch_type(spec)
            batch_size = len(parse_batch_spec(spec))
            row = [spec, batch_type, str(batch_size)]
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


def is_mla_backend(backend: str) -> bool:
    """
    Check if backend is an MLA backend using the AttentionBackendEnum.

    Args:
        backend: Backend name matching AttentionBackendEnum exactly
        (e.g., "FLASHMLA_SPARSE")

    Returns:
        True if the backend is an MLA backend, False otherwise
    """
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    try:
        backend_enum = AttentionBackendEnum[backend]
        backend_class = backend_enum.get_class()
        return backend_class.is_mla()
    except (KeyError, ValueError, ImportError, AttributeError):
        return False

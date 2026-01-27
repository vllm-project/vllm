# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Standard attention benchmark runner - shared utilities for non-MLA benchmarks.

This module provides helpers for running standard attention backends
(FlashAttention, Triton, FlashInfer) with real vLLM integration.
"""

import types

import numpy as np
import torch
from batch_spec import parse_batch_spec, reorder_for_flashinfer
from common import BenchmarkConfig, BenchmarkResult, MockLayer, get_attention_scale

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
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import FullAttentionSpec

# ============================================================================
# Backend Configuration
# ============================================================================


_BACKEND_CONFIG = {
    "flash": {
        "module": "vllm.v1.attention.backends.flash_attn",
        "backend_class": "FlashAttentionBackend",
        "dtype": torch.float16,
        "cache_layout": "standard",
        # ^ [2, num_blocks, block_size, num_kv_heads, head_dim]
    },
    "triton": {
        "module": "vllm.v1.attention.backends.triton_attn",
        "backend_class": "TritonAttentionBackend",
        "dtype": torch.float32,
        "cache_layout": "standard",
    },
    "flashinfer": {
        "module": "vllm.v1.attention.backends.flashinfer",
        "backend_class": "FlashInferBackend",
        "dtype": torch.float16,
        "cache_layout": "flashinfer",
        # ^ [num_blocks, 2, block_size, num_kv_heads, head_dim]
    },
}


def _get_backend_config(backend: str) -> dict:
    if backend not in _BACKEND_CONFIG:
        raise ValueError(
            f"Unknown backend: {backend}. "
            f"Available: {', '.join(_BACKEND_CONFIG.keys())}"
        )
    return _BACKEND_CONFIG[backend]


# ============================================================================
# Metadata Building Helpers
# ============================================================================


def _build_common_attn_metadata(
    q_lens: list[int],
    kv_lens: list[int],
    block_size: int,
    device: torch.device,
) -> CommonAttentionMetadata:
    """Build CommonAttentionMetadata from query/kv lengths."""
    batch_size = len(q_lens)
    total_tokens = sum(q_lens)

    query_start_loc = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    query_start_loc[1:] = torch.tensor(q_lens, dtype=torch.int32, device=device).cumsum(
        0
    )
    query_start_loc_cpu = query_start_loc.cpu()

    seq_lens = torch.tensor(kv_lens, dtype=torch.int32, device=device)
    seq_lens_cpu = seq_lens.cpu()
    max_seq_len = int(seq_lens_cpu.max())

    context_lens = [kv - q for kv, q in zip(kv_lens, q_lens)]
    num_computed_tokens_cpu = torch.tensor(context_lens, dtype=torch.int32)

    max_blocks = (max(kv_lens) + block_size - 1) // block_size
    num_blocks = batch_size * max_blocks
    block_table_tensor = torch.arange(
        num_blocks, dtype=torch.int32, device=device
    ).view(batch_size, max_blocks)
    slot_mapping = torch.arange(total_tokens, dtype=torch.int64, device=device)

    max_query_len = max(q_lens)

    return CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens_cpu,
        num_computed_tokens_cpu=num_computed_tokens_cpu,
        num_reqs=batch_size,
        num_actual_tokens=total_tokens,
        max_query_len=max_query_len,
        max_seq_len=max_seq_len,
        block_table_tensor=block_table_tensor,
        slot_mapping=slot_mapping,
        causal=True,
    )


def _create_vllm_config(
    config: BenchmarkConfig,
    dtype: torch.dtype,
    max_num_blocks: int,
) -> VllmConfig:
    """Create a VllmConfig for benchmarking with mock model methods."""
    model_config = ModelConfig(
        model="meta-llama/Meta-Llama-3-8B",
        tokenizer="meta-llama/Meta-Llama-3-8B",
        trust_remote_code=False,
        dtype=dtype,
        seed=0,
        max_model_len=1024,
    )

    cache_config = CacheConfig(
        block_size=config.block_size,
        cache_dtype="auto",
        swap_space=0,
    )
    cache_config.num_gpu_blocks = max_num_blocks
    cache_config.num_cpu_blocks = 0

    parallel_config = ParallelConfig(tensor_parallel_size=1)
    scheduler_config = SchedulerConfig(
        max_num_seqs=256,
        max_num_batched_tokens=8192,
        enable_chunked_prefill=True,
    )
    device_config = DeviceConfig()
    load_config = LoadConfig()
    compilation_config = CompilationConfig()

    # Add mock methods for benchmark config values
    model_config.get_num_layers = types.MethodType(
        lambda self: config.num_layers, model_config
    )
    model_config.get_sliding_window_for_layer = types.MethodType(
        lambda self, i: None, model_config
    )
    model_config.get_logits_soft_cap_for_layer = types.MethodType(
        lambda self, i: 0.0, model_config
    )
    model_config.get_sm_scale_for_layer = types.MethodType(
        lambda self, i: 1.0 / config.head_dim**0.5, model_config
    )
    model_config.get_num_attention_heads = types.MethodType(
        lambda self, parallel_config=None: config.num_q_heads, model_config
    )
    model_config.get_num_kv_heads = types.MethodType(
        lambda self, parallel_config=None: config.num_kv_heads, model_config
    )
    model_config.get_head_size = types.MethodType(
        lambda self: config.head_dim, model_config
    )
    model_config.get_sliding_window = types.MethodType(lambda self: None, model_config)

    return VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=parallel_config,
        scheduler_config=scheduler_config,
        device_config=device_config,
        load_config=load_config,
        compilation_config=compilation_config,
    )


# ============================================================================
# Backend Initialization
# ============================================================================


def _create_backend_impl(
    backend_cfg: dict,
    config: BenchmarkConfig,
    device: torch.device,
):
    """Create backend implementation instance."""
    import importlib

    backend_module = importlib.import_module(backend_cfg["module"])
    backend_class = getattr(backend_module, backend_cfg["backend_class"])

    scale = get_attention_scale(config.head_dim)
    dtype = backend_cfg["dtype"]

    impl = backend_class.get_impl_cls()(
        num_heads=config.num_q_heads,
        head_size=config.head_dim,
        scale=scale,
        num_kv_heads=config.num_kv_heads,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
    )

    kv_cache_spec = FullAttentionSpec(
        block_size=config.block_size,
        num_kv_heads=config.num_kv_heads,
        head_size=config.head_dim,
        dtype=dtype,
    )

    layer = MockLayer(device, kv_cache_spec=kv_cache_spec)

    return backend_class, impl, layer, dtype


def _create_metadata_builder(
    backend_class,
    kv_cache_spec: FullAttentionSpec,
    vllm_config: VllmConfig,
    device: torch.device,
):
    """Create metadata builder instance."""
    return backend_class.get_builder_cls()(
        kv_cache_spec=kv_cache_spec,
        layer_names=["layer_0"],
        vllm_config=vllm_config,
        device=device,
    )


# ============================================================================
# Tensor Creation Helpers
# ============================================================================


def _create_input_tensors(
    config: BenchmarkConfig,
    total_q: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple:
    """Create Q, K, V input tensors for all layers."""
    q_list = [
        torch.randn(
            total_q, config.num_q_heads, config.head_dim, device=device, dtype=dtype
        )
        for _ in range(config.num_layers)
    ]
    k_list = [
        torch.randn(
            total_q, config.num_kv_heads, config.head_dim, device=device, dtype=dtype
        )
        for _ in range(config.num_layers)
    ]
    v_list = [
        torch.randn(
            total_q, config.num_kv_heads, config.head_dim, device=device, dtype=dtype
        )
        for _ in range(config.num_layers)
    ]
    return q_list, k_list, v_list


def _create_kv_cache(
    config: BenchmarkConfig,
    max_num_blocks: int,
    cache_layout: str,
    device: torch.device,
    dtype: torch.dtype,
) -> list:
    """Create KV cache tensors for all layers."""
    if cache_layout == "flashinfer":
        # FlashInfer layout: [num_blocks, 2, block_size, num_kv_heads, head_dim]
        cache_list = [
            torch.zeros(
                max_num_blocks,
                2,
                config.block_size,
                config.num_kv_heads,
                config.head_dim,
                device=device,
                dtype=dtype,
            )
            for _ in range(config.num_layers)
        ]
    else:
        # Standard layout: [2, num_blocks, block_size, num_kv_heads, head_dim]
        cache_list = [
            torch.zeros(
                2,
                max_num_blocks,
                config.block_size,
                config.num_kv_heads,
                config.head_dim,
                device=device,
                dtype=dtype,
            )
            for _ in range(config.num_layers)
        ]
    return cache_list


# ============================================================================
# Benchmark Execution
# ============================================================================


def _run_single_benchmark(
    config: BenchmarkConfig,
    impl,
    layer,
    q_list: list,
    k_list: list,
    v_list: list,
    cache_list: list,
    attn_metadata,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple:
    """Run single benchmark iteration with warmup and timing loop."""
    total_q = q_list[0].shape[0]
    out = torch.empty(
        total_q, config.num_q_heads, config.head_dim, device=device, dtype=dtype
    )

    # Warmup
    for _ in range(config.warmup_iters):
        for i in range(config.num_layers):
            impl.forward(
                layer,
                q_list[i],
                k_list[i],
                v_list[i],
                cache_list[i],
                attn_metadata,
                output=out,
            )
    torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(config.repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for i in range(config.num_layers):
            impl.forward(
                layer,
                q_list[i],
                k_list[i],
                v_list[i],
                cache_list[i],
                attn_metadata,
                output=out,
            )
        end.record()

        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end)
        times.append(elapsed_ms / 1000.0 / config.num_layers)  # seconds per layer

    mem_stats = {}
    if config.profile_memory:
        mem_stats = {
            "allocated_mb": torch.cuda.memory_allocated(device) / 1024**2,
            "reserved_mb": torch.cuda.memory_reserved(device) / 1024**2,
        }

    return times, mem_stats


# ============================================================================
# Public API
# ============================================================================


def run_attention_benchmark(config: BenchmarkConfig) -> BenchmarkResult:
    """
    Run standard attention benchmark with real kernels.

    Supports: flash, triton, flashinfer

    Args:
        config: Benchmark configuration

    Returns:
        BenchmarkResult with timing and memory statistics
    """
    device = torch.device(config.device)
    torch.cuda.set_device(device)

    backend_cfg = _get_backend_config(config.backend)

    requests = parse_batch_spec(config.batch_spec)

    if config.backend == "flashinfer":
        requests = reorder_for_flashinfer(requests)

    q_lens = [r.q_len for r in requests]
    kv_lens = [r.kv_len for r in requests]
    total_q = sum(q_lens)
    max_kv = max(kv_lens)

    max_num_blocks = (max_kv + config.block_size - 1) // config.block_size

    backend_class, impl, layer, dtype = _create_backend_impl(
        backend_cfg, config, device
    )

    common_metadata = _build_common_attn_metadata(
        q_lens, kv_lens, config.block_size, device
    )

    kv_cache_spec = FullAttentionSpec(
        block_size=config.block_size,
        num_kv_heads=config.num_kv_heads,
        head_size=config.head_dim,
        dtype=dtype,
    )

    vllm_config = _create_vllm_config(config, dtype, max_num_blocks)

    builder = _create_metadata_builder(
        backend_class, kv_cache_spec, vllm_config, device
    )

    attn_metadata = builder.build(
        common_prefix_len=0,
        common_attn_metadata=common_metadata,
    )

    q_list, k_list, v_list = _create_input_tensors(config, total_q, device, dtype)

    cache_list = _create_kv_cache(
        config, max_num_blocks, backend_cfg["cache_layout"], device, dtype
    )

    times, mem_stats = _run_single_benchmark(
        config,
        impl,
        layer,
        q_list,
        k_list,
        v_list,
        cache_list,
        attn_metadata,
        device,
        dtype,
    )

    mean_time = np.mean(times)
    throughput = total_q / mean_time if mean_time > 0 else 0

    return BenchmarkResult(
        config=config,
        mean_time=mean_time,
        std_time=np.std(times),
        min_time=np.min(times),
        max_time=np.max(times),
        throughput_tokens_per_sec=throughput,
        memory_allocated_mb=mem_stats.get("allocated_mb"),
        memory_reserved_mb=mem_stats.get("reserved_mb"),
    )

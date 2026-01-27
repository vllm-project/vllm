# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Standard attention benchmark runner - shared utilities for non-MLA benchmarks.

This module provides helpers for running standard attention backends
(FlashAttention, Triton, FlashInfer) with real vLLM integration.
"""

import numpy as np
import torch
from batch_spec import parse_batch_spec, reorder_for_flashinfer
from common import BenchmarkConfig, BenchmarkResult, MockLayer, get_attention_scale

from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.worker.block_table import BlockTable

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
    """
    Get backend configuration.

    Args:
        backend: Backend name (flash, triton, flashinfer)

    Returns:
        Backend configuration dict

    Raises:
        ValueError: If backend is unknown
    """
    if backend not in _BACKEND_CONFIG:
        raise ValueError(
            f"Unknown backend: {backend}. "
            f"Available: {', '.join(_BACKEND_CONFIG.keys())}"
        )
    return _BACKEND_CONFIG[backend]


# ============================================================================
# Metadata Building Helpers
# ============================================================================


def _build_attention_metadata(
    requests: list,
    block_size: int,
    device: torch.device,
) -> tuple:
    """
    Build CommonAttentionMetadata from batch requests.

    Args:
        requests: List of BatchRequest objects
        block_size: KV cache block size
        device: Target device

    Returns:
        Tuple of (metadata, slot_mapping, max_num_blocks)
    """
    q_lens = [r.q_len for r in requests]
    kv_lens = [r.kv_len for r in requests]
    total_q = sum(q_lens)
    max_kv = max(kv_lens)

    # Build query start locations
    q_start_cpu = np.array(
        [0] + [sum(q_lens[: i + 1]) for i in range(len(q_lens))],
        dtype=np.int32,
    )
    q_start_gpu = torch.from_numpy(q_start_cpu).to(device)

    # Build sequence lengths
    seq_lens_cpu = np.array(kv_lens, dtype=np.int32)
    seq_lens_gpu = torch.from_numpy(seq_lens_cpu).to(device)

    # Build num_computed_tokens (context length before new query)
    computed_tokens_cpu = np.array(
        [kv - q for kv, q in zip(kv_lens, q_lens)],
        dtype=np.int32,
    )

    # Build block table
    num_blocks_per_req = [(kv + block_size - 1) // block_size for kv in kv_lens]
    max_num_blocks = max(num_blocks_per_req)

    block_table_cpu = np.zeros((len(requests), max_num_blocks), dtype=np.int32)
    for i, num_blocks in enumerate(num_blocks_per_req):
        block_table_cpu[i, :num_blocks] = np.arange(num_blocks, dtype=np.int32)
    block_table_gpu = torch.from_numpy(block_table_cpu).to(device)

    # Build slot mapping (maps each token to its KV cache slot)
    slot_mapping_list = []
    for i, (q_len, kv_len, num_blocks) in enumerate(
        zip(q_lens, kv_lens, num_blocks_per_req)
    ):
        context_len = kv_len - q_len
        for j in range(q_len):
            token_kv_idx = context_len + j
            block_idx = token_kv_idx // block_size
            offset_in_block = token_kv_idx % block_size
            global_block_id = block_table_cpu[i, block_idx]
            slot_id = global_block_id * block_size + offset_in_block
            slot_mapping_list.append(slot_id)

    slot_mapping = torch.tensor(slot_mapping_list, dtype=torch.int64, device=device)

    # Create CommonAttentionMetadata
    metadata = CommonAttentionMetadata(
        query_start_loc=q_start_gpu,
        query_start_loc_cpu=torch.from_numpy(q_start_cpu),
        seq_lens=seq_lens_gpu,
        seq_lens_cpu=torch.from_numpy(seq_lens_cpu),
        num_computed_tokens_cpu=torch.from_numpy(computed_tokens_cpu),
        num_reqs=len(requests),
        num_actual_tokens=total_q,
        max_query_len=max(q_lens),
        max_seq_len=max_kv,
        block_table_tensor=block_table_gpu,
        slot_mapping=slot_mapping,
    )

    return metadata, slot_mapping, max_num_blocks


def _build_block_table(
    requests: list,
    kv_lens: list[int],
    block_size: int,
    total_q: int,
    max_num_blocks: int,
    device: torch.device,
) -> BlockTable:
    """
    Build BlockTable for metadata builder.

    Args:
        requests: List of BatchRequest objects
        kv_lens: List of KV sequence lengths
        block_size: KV cache block size
        total_q: Total number of query tokens
        max_num_blocks: Maximum number of blocks per request
        device: Target device

    Returns:
        BlockTable instance
    """
    bt = BlockTable(
        block_size=block_size,
        max_num_reqs=len(requests),
        max_num_blocks_per_req=max_num_blocks,
        max_num_batched_tokens=total_q,
        pin_memory=False,
        device=device,
        kernel_block_size=block_size,
        cp_kv_cache_interleave_size=1,
    )
    for i in range(len(requests)):
        num_blocks = (kv_lens[i] + block_size - 1) // block_size
        bt.add_row(list(range(num_blocks)), i)
    bt.commit(len(requests))
    return bt


# ============================================================================
# Backend Initialization
# ============================================================================


def _create_backend_impl(
    backend_cfg: dict,
    config: BenchmarkConfig,
    device: torch.device,
):
    """
    Create backend implementation instance.

    Args:
        backend_cfg: Backend configuration dict
        config: BenchmarkConfig instance
        device: Target device

    Returns:
        Tuple of (backend_class, impl, layer, dtype)
    """
    # Import backend class
    import importlib

    backend_module = importlib.import_module(backend_cfg["module"])
    backend_class = getattr(backend_module, backend_cfg["backend_class"])

    # Calculate scale
    scale = get_attention_scale(config.head_dim)

    # Get dtype
    dtype = backend_cfg["dtype"]

    # Create attention impl
    impl = backend_class.get_impl_cls()(
        num_heads=config.num_q_heads,
        head_size=config.head_dim,
        scale=scale,
        num_kv_heads=config.num_kv_heads,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
    )

    # Create KV cache spec for MockLayer
    from vllm.v1.kv_cache_interface import AttentionSpec

    kv_cache_spec = AttentionSpec(
        block_size=config.block_size,
        num_kv_heads=config.num_kv_heads,
        head_size=config.head_dim,
        dtype=dtype,
    )

    # Create mock layer
    layer = MockLayer(device, kv_cache_spec=kv_cache_spec)

    return backend_class, impl, layer, dtype


def _create_metadata_builder(
    backend_class,
    common_metadata: CommonAttentionMetadata,
    block_table: BlockTable,
    config: BenchmarkConfig,
    dtype: torch.dtype,
    device: torch.device,
):
    """
    Create metadata builder instance.

    Args:
        backend_class: Backend class
        common_metadata: CommonAttentionMetadata instance
        block_table: BlockTable instance
        config: BenchmarkConfig instance
        dtype: Tensor dtype
        device: Target device

    Returns:
        Built attention metadata
    """
    # Create mock runner for builder
    from common import MockRunner

    runner = MockRunner(
        seq_lens=common_metadata.seq_lens_cpu.numpy(),
        query_start_locs=common_metadata.query_start_loc_cpu.numpy(),
        device=device,
        num_q_heads=config.num_q_heads,
        num_kv_heads=config.num_kv_heads,
        head_dim=config.head_dim,
        dtype=dtype,
    )

    # Create metadata builder
    builder = backend_class.get_builder_cls()(
        runner=runner,
        kv_cache_spec=AttentionSpec(
            block_size=config.block_size,
            num_kv_heads=config.num_kv_heads,
            head_size=config.head_dim,
            dtype=dtype,
        ),
        block_table=block_table,
    )

    return builder


# ============================================================================
# Tensor Creation Helpers
# ============================================================================


def _create_input_tensors(
    config: BenchmarkConfig,
    total_q: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple:
    """
    Create Q, K, V input tensors for all layers.

    Args:
        config: BenchmarkConfig instance
        total_q: Total number of query tokens
        device: Target device
        dtype: Tensor dtype

    Returns:
        Tuple of (q_list, k_list, v_list)
    """
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
    """
    Create KV cache tensors for all layers.

    Args:
        config: BenchmarkConfig instance
        max_num_blocks: Maximum number of blocks
        cache_layout: Cache layout type ("standard" or "flashinfer")
        device: Target device
        dtype: Tensor dtype

    Returns:
        List of KV cache tensors (one per layer)
    """
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
    """
    Run single benchmark iteration with warmup and timing loop.

    Args:
        config: BenchmarkConfig instance
        impl: Backend implementation instance
        layer: MockLayer instance
        q_list: List of Q tensors
        k_list: List of K tensors
        v_list: List of V tensors
        cache_list: List of KV cache tensors
        attn_metadata: Attention metadata
        device: Target device
        dtype: Tensor dtype

    Returns:
        Tuple of (times, mem_stats)
    """
    # Create output buffer
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

    # Memory stats
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

    # Get backend configuration
    backend_cfg = _get_backend_config(config.backend)

    # Parse batch spec
    requests = parse_batch_spec(config.batch_spec)

    # Reorder for FlashInfer if needed
    if config.backend == "flashinfer":
        requests = reorder_for_flashinfer(requests)

    # Extract dimensions
    q_lens = [r.q_len for r in requests]
    kv_lens = [r.kv_len for r in requests]
    total_q = sum(q_lens)

    # Build common metadata
    common_metadata, slot_mapping, max_num_blocks = _build_attention_metadata(
        requests, config.block_size, device
    )

    # Create backend impl, layer, and dtype
    backend_class, impl, layer, dtype = _create_backend_impl(
        backend_cfg, config, device
    )

    # Build block table
    block_table = _build_block_table(
        requests, kv_lens, config.block_size, total_q, max_num_blocks, device
    )

    # Create metadata builder and build metadata
    builder = _create_metadata_builder(
        backend_class, common_metadata, block_table, config, dtype, device
    )

    attn_metadata = builder.build(
        num_reqs=len(requests),
        num_actual_tokens=total_q,
        max_query_len=max(q_lens),
        common_prefix_len=0,
        common_attn_metadata=common_metadata,
    )

    # Create input tensors
    q_list, k_list, v_list = _create_input_tensors(config, total_q, device, dtype)

    # Create KV cache
    cache_list = _create_kv_cache(
        config, max_num_blocks, backend_cfg["cache_layout"], device, dtype
    )

    # Run benchmark
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

    # Calculate throughput
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


# ============================================================================
# Backwards Compatibility
# ============================================================================


# Keep old function names for backwards compatibility
def build_common_metadata(*args, **kwargs):
    """Deprecated: Use _build_attention_metadata instead."""
    return _build_attention_metadata(*args, **kwargs)


def run_attention_benchmark_impl(config: BenchmarkConfig) -> BenchmarkResult:
    """Deprecated: Use run_attention_benchmark instead."""
    return run_attention_benchmark(config)


def run_mla_benchmark_impl(config: BenchmarkConfig) -> BenchmarkResult:
    """
    Run MLA benchmark with real kernels.

    This is a stub - use mla_runner.py for MLA benchmarks.
    """
    raise NotImplementedError(
        "MLA benchmark runner is in mla_runner.py. "
        "Use run_mla_benchmark() from that module."
    )

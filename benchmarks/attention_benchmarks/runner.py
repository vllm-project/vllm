# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Complete benchmark runner with real vLLM integration.

This module provides working implementations that can actually run
attention kernels, not placeholders.
"""

import numpy as np
import torch
from batch_spec import BatchRequest, parse_batch_spec, reorder_for_flashinfer
from common import (
    BenchmarkConfig,
    BenchmarkResult,
    MockLayer,
    MockRunner,
    get_attention_scale,
)

from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.worker.block_table import BlockTable


def build_common_metadata(
    requests: list[BatchRequest],
    block_size: int,
    device: torch.device,
) -> tuple[CommonAttentionMetadata, torch.Tensor, int]:
    """
    Build CommonAttentionMetadata from batch requests.

    Args:
        requests: List of BatchRequest
        block_size: KV cache block size
        device: Torch device

    Returns:
        Tuple of (CommonAttentionMetadata, slot_mapping, max_num_blocks)
    """
    q_lens = [r.q_len for r in requests]
    kv_lens = [r.kv_len for r in requests]
    total_q = sum(q_lens)
    max_kv = max(kv_lens)

    # Build query start locations
    q_start_cpu = np.array(
        [0] + [sum(q_lens[: i + 1]) for i in range(len(q_lens))], dtype=np.int32
    )
    q_start_gpu = torch.from_numpy(q_start_cpu).to(device)

    # Build sequence lengths
    seq_lens_cpu = np.array(kv_lens, dtype=np.int32)
    seq_lens_gpu = torch.from_numpy(seq_lens_cpu).to(device)

    # Computed tokens (context before new query)
    computed_tokens_cpu = np.array(
        [kv - q for kv, q in zip(kv_lens, q_lens)], dtype=np.int32
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
        # For each token in the query, map to its slot in the KV cache
        context_len = kv_len - q_len
        for j in range(q_len):
            token_kv_idx = context_len + j
            block_idx = token_kv_idx // block_size
            offset_in_block = token_kv_idx % block_size
            # Global slot ID
            global_block_id = block_table_cpu[i, block_idx]
            slot_id = global_block_id * block_size + offset_in_block
            slot_mapping_list.append(slot_id)

    slot_mapping = torch.tensor(slot_mapping_list, dtype=torch.int64, device=device)

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


def run_attention_benchmark_impl(config: BenchmarkConfig) -> BenchmarkResult:
    """
    Run standard attention benchmark with real kernels.

    Args:
        config: Benchmark configuration

    Returns:
        BenchmarkResult with actual timing data
    """
    device = torch.device(config.device)
    torch.cuda.set_device(device)

    # Parse batch spec
    requests = parse_batch_spec(config.batch_spec)

    # Reorder for FlashInfer if needed
    if config.backend == "flashinfer":
        requests = reorder_for_flashinfer(requests)

    # Extract dimensions
    q_lens = [r.q_len for r in requests]
    kv_lens = [r.kv_len for r in requests]
    total_q = sum(q_lens)

    # Compute scale
    scale = get_attention_scale(config.head_dim)

    # Select backend and dtype
    if config.backend == "flash":
        from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend as BE

        dt = torch.float16
    elif config.backend == "triton":
        from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend as BE

        dt = torch.float32
    elif config.backend == "flashinfer":
        from vllm.v1.attention.backends.flashinfer import FlashInferBackend as BE

        dt = torch.float16
    else:
        raise ValueError(f"Unknown backend: {config.backend}")

    # Create attention impl
    impl = BE.get_impl_cls()(
        num_heads=config.num_q_heads,
        head_size=config.head_dim,
        scale=scale,
        num_kv_heads=config.num_kv_heads,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
    )

    layer = MockLayer(device)

    # Build metadata
    common_metadata, slot_mapping, max_num_blocks = build_common_metadata(
        requests, config.block_size, device
    )

    # Create mock runner for builder
    runner = MockRunner(
        seq_lens=common_metadata.seq_lens_cpu.numpy(),
        query_start_locs=common_metadata.query_start_loc_cpu.numpy(),
        device=device,
        num_q_heads=config.num_q_heads,
        num_kv_heads=config.num_kv_heads,
        head_dim=config.head_dim,
        dtype=dt,
    )

    # Build block table
    bt = BlockTable(len(requests), max_num_blocks, total_q, False, device)
    for i in range(len(requests)):
        num_blocks = (kv_lens[i] + config.block_size - 1) // config.block_size
        bt.add_row(list(range(num_blocks)), i)
    bt.commit(len(requests))

    # Create metadata builder
    builder = BE.get_builder_cls()(
        runner=runner,
        kv_cache_spec=AttentionSpec(
            block_size=config.block_size,
            num_kv_heads=config.num_kv_heads,
            head_size=config.head_dim,
            dtype=dt,
            use_mla=False,
        ),
        block_table=bt,
    )

    # Build attention metadata
    attn_metadata = builder.build(
        num_reqs=len(requests),
        num_actual_tokens=total_q,
        max_query_len=max(q_lens),
        common_prefix_len=0,
        common_attn_metadata=common_metadata,
    )

    # Create input tensors
    q_list = [
        torch.randn(
            total_q, config.num_q_heads, config.head_dim, device=device, dtype=dt
        )
        for _ in range(config.num_layers)
    ]
    k_list = [
        torch.randn(
            total_q, config.num_kv_heads, config.head_dim, device=device, dtype=dt
        )
        for _ in range(config.num_layers)
    ]
    v_list = [
        torch.randn(
            total_q, config.num_kv_heads, config.head_dim, device=device, dtype=dt
        )
        for _ in range(config.num_layers)
    ]

    # KV cache
    if config.backend == "flashinfer":
        cache_list = [
            torch.zeros(
                max_num_blocks,
                2,
                config.block_size,
                config.num_kv_heads,
                config.head_dim,
                device=device,
                dtype=dt,
            )
            for _ in range(config.num_layers)
        ]
    else:
        cache_list = [
            torch.zeros(
                2,
                max_num_blocks,
                config.block_size,
                config.num_kv_heads,
                config.head_dim,
                device=device,
                dtype=dt,
            )
            for _ in range(config.num_layers)
        ]

    # Output buffer
    out = torch.empty(
        total_q, config.num_q_heads, config.head_dim, device=device, dtype=dt
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

    # Throughput
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


def run_mla_benchmark_impl(config: BenchmarkConfig) -> BenchmarkResult:
    """
    Run MLA benchmark with real kernels.

    This is a template - needs specific backend implementation.
    """
    # TODO: Implement for specific MLA backends
    # This requires more complex setup due to MLA-specific metadata
    raise NotImplementedError(
        "MLA benchmark runner needs backend-specific implementation. "
        "See benchmark_mla_numsplits.py for CUTLASS MLA example."
    )

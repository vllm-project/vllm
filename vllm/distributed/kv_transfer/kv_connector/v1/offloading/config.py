# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Translate vLLM KV cache metadata for native offloading backends."""

from typing import TYPE_CHECKING

from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes
from vllm.v1.kv_cache_interface import FullAttentionSpec, MLAAttentionSpec
from vllm.v1.kv_offload.config import (
    CompactGroupSliceConfig,
    CompactSliceConfig,
    OffloadingCacheConfig,
    OffloadingConfig,
    OffloadingGroupConfig,
    OffloadingModelConfig,
    OffloadingParallelConfig,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheTensor


def is_kv_cache_tensor_packed(kv_cache_tensor: "KVCacheTensor") -> bool:
    """Return whether a KV cache tensor uses a packed block stride."""
    return bool(kv_cache_tensor.block_stride)


def _build_compact_slice_accounting(
    kv_cache_config: "KVCacheConfig",
    vllm_config: "VllmConfig",
) -> tuple[CompactGroupSliceConfig, ...] | None:
    """Derive per-slice accounting from packed tensor metadata.

    Returns None when the config lacks packed tensor metadata or the
    compact layout is not enabled.  Requires the compact aggregate signature
    (from the scheduler-side pre-projection global groups) plus packed tensor
    metadata (from the post-projection worker config); when both are present
    the slice accounting is derived regardless of whether
    ``build_offloading_config`` is called on the scheduler or worker side.
    """
    signature = kv_cache_config.compact_aggregate_signature
    if signature is None:
        return None

    # Need packed tensor metadata for full slice accounting.
    if not kv_cache_config.kv_cache_tensors or not any(
        t.block_stride for t in kv_cache_config.kv_cache_tensors
    ):
        return None

    # Lazy import to keep cpu module out of the pure config path.
    # build_compact_layout_accounting requires packed tensors.
    from vllm.v1.kv_offload.cpu.compact_accounting import (
        build_compact_layout_accounting,
    )

    world_size = vllm_config.parallel_config.world_size
    # Use a sentinel cpu_budget_bytes — GroupCompactAccounting derives its
    # slice fields from packed tensor geometry, not the CPU budget.
    cpu_budget_bytes = 1
    accounting = build_compact_layout_accounting(
        kv_cache_config,
        world_size=world_size,
        block_size_factor=1,
        cpu_budget_bytes=cpu_budget_bytes,
    )

    group_slices: list[CompactGroupSliceConfig] = []
    for group in accounting.groups:
        slices = tuple(
            CompactSliceConfig(
                offset_bytes=s.offset_bytes,
                real_bytes_per_gpu_block=s.real_bytes_per_gpu_block,
                padded_bytes_per_gpu_block=s.padded_bytes_per_gpu_block,
                layer_name=s.layer_name,
            )
            for s in group.slices
        )
        group_slices.append(
            CompactGroupSliceConfig(
                group_idx=group.group_idx,
                slices=slices,
                compact_real_bytes_per_rank=group.compact_real_bytes_per_rank,
                compact_padded_bytes_per_rank=group.compact_padded_bytes_per_rank,
            )
        )

    # Validate aggregate slice accounting equals transported group charges.
    for gs in group_slices:
        charge = signature[gs.group_idx]
        assert (
            gs.compact_real_bytes_per_rank
            == charge.compact_bytes_per_native_block_per_worker
        ), (
            f"Group {gs.group_idx} slice accounting "
            f"{gs.compact_real_bytes_per_rank} != transported charge "
            f"{charge.compact_bytes_per_native_block_per_worker}"
        )

    return tuple(group_slices)


def build_offloading_config(
    vllm_config: "VllmConfig",
    kv_cache_config: "KVCacheConfig",
) -> OffloadingConfig:
    """Translate vLLM configuration into the native offloading boundary."""
    kv_transfer_config = vllm_config.kv_transfer_config
    assert kv_transfer_config is not None
    extra_config = kv_transfer_config.kv_connector_extra_config
    assert kv_transfer_config.engine_id is not None
    engine_id = kv_transfer_config.engine_id

    parallel_config = vllm_config.parallel_config
    context_parallel_factor = (
        parallel_config.decode_context_parallel_size
        * parallel_config.prefill_context_parallel_size
    )

    # Build groups, optionally populating compact_bytes_per_worker from
    # the transported signature.
    signature = kv_cache_config.compact_aggregate_signature
    groups = []

    # Validate group count matches signature before any positional access.
    if signature is not None:
        assert len(kv_cache_config.kv_cache_groups) == len(signature), (
            f"Group count {len(kv_cache_config.kv_cache_groups)} "
            f"does not match signature count {len(signature)}"
        )

    for group_idx, group in enumerate(kv_cache_config.kv_cache_groups):
        compact_bytes = None
        if signature is not None:
            charge = signature[group_idx]
            assert charge.group_idx == group_idx, (
                f"Signature group index {charge.group_idx} "
                f"does not match builder group index {group_idx}"
            )
            compact_bytes = charge.compact_bytes_per_native_block_per_worker
        groups.append(
            OffloadingGroupConfig(
                tokens_per_block=(
                    group.kv_cache_spec.block_size * context_parallel_factor
                ),
                layer_names=tuple(group.layer_names),
                compact_bytes_per_native_block_per_worker=compact_bytes,
            )
        )
    groups = tuple(groups)

    _, tokens_per_hash = resolve_kv_cache_block_sizes(kv_cache_config, vllm_config)
    for group in groups:
        assert group.tokens_per_block % tokens_per_hash == 0, (
            f"tokens_per_block={group.tokens_per_block} not divisible by "
            f"tokens_per_hash={tokens_per_hash}. "
            f"Hybrid models (e.g. Mamba+Attention) need "
            f"--enable-prefix-caching to align block sizes."
        )

    blocks_per_chunk = 1
    blocks_per_chunk_config = extra_config.get("blocks_per_chunk")
    tokens_per_chunk = extra_config.get("block_size")

    if blocks_per_chunk_config is not None and tokens_per_chunk is not None:
        raise ValueError(
            "Specify only one of 'block_size' or 'blocks_per_chunk' "
            "in kv_connector_extra_config."
        )

    if blocks_per_chunk_config is not None:
        blocks_per_chunk = int(blocks_per_chunk_config)

        if blocks_per_chunk <= 0:
            raise ValueError("'blocks_per_chunk' must be greater than 0.")

    elif tokens_per_chunk is not None:
        tokens_per_chunk_int = int(tokens_per_chunk)

        unique_tokens_per_block = {group.tokens_per_block for group in groups}

        assert len(unique_tokens_per_block) == 1, (
            "If 'block_size' is specified in kv_connector_extra_config, "
            "there must be at least one KV cache group, "
            "and all groups must have the same block size."
        )

        tokens_per_block = unique_tokens_per_block.pop()
        assert tokens_per_chunk_int % tokens_per_block == 0
        blocks_per_chunk = tokens_per_chunk_int // tokens_per_block

    worker_kv_bytes_per_block = 0
    if kv_cache_config.num_blocks > 0:
        packed_tensors = tuple(
            is_kv_cache_tensor_packed(tensor)
            for tensor in kv_cache_config.kv_cache_tensors
        )
        is_packed = any(packed_tensors)
        assert not is_packed or all(packed_tensors)
        total_gpu_kv_bytes = (
            kv_cache_config.kv_cache_tensors[0].size
            if is_packed
            else sum(tensor.size for tensor in kv_cache_config.kv_cache_tensors)
        )
        worker_kv_bytes_per_block = total_gpu_kv_bytes // kv_cache_config.num_blocks

    # Only a single non-MLA full-attention group is parallelism-invariant:
    # MLA latent KV is replicated per rank (never head-sharded), and the V2
    # model runner's KV layout is not known to be parallelism-invariant.
    single_group = (
        kv_cache_config.kv_cache_groups[0].kv_cache_spec
        if len(kv_cache_config.kv_cache_groups) == 1
        else None
    )
    is_parallelism_agnostic = (
        not vllm_config.use_v2_model_runner
        and single_group is not None
        and isinstance(single_group, FullAttentionSpec)
        and not isinstance(single_group, MLAAttentionSpec)
    )

    # Derive compact slice accounting from the transported signature and
    # packed tensor metadata when both are available.  build_offloading_config
    # runs on both the scheduler (pre-projection, global groups) and worker
    # (post-projection) side; when the scheduler-side config has packed
    # tensor metadata the slice accounting is derived there too.
    compact_slice_accounting = _build_compact_slice_accounting(
        kv_cache_config, vllm_config
    )

    kv_events_config = vllm_config.kv_events_config
    return OffloadingConfig(
        groups=groups,
        worker_kv_bytes_per_block=worker_kv_bytes_per_block,
        enable_kv_cache_events=(
            kv_events_config is not None and kv_events_config.enable_kv_cache_events
        ),
        extra_config=extra_config,
        engine_id=engine_id,
        model=OffloadingModelConfig(
            name=vllm_config.model_config.model,
            dtype=str(vllm_config.cache_config.cache_dtype).replace("torch.", ""),
        ),
        cache=OffloadingCacheConfig(
            tokens_per_hash=tokens_per_hash,
            blocks_per_chunk=blocks_per_chunk,
        ),
        parallel=OffloadingParallelConfig(
            rank=parallel_config.rank,
            world_size=parallel_config.world_size,
            tp_size=parallel_config.tensor_parallel_size,
            pp_size=parallel_config.pipeline_parallel_size,
            pcp_size=parallel_config.prefill_context_parallel_size,
            dcp_size=parallel_config.decode_context_parallel_size,
            data_parallel_index=parallel_config.data_parallel_index,
            is_parallelism_agnostic=is_parallelism_agnostic,
        ),
        compact_slice_accounting=compact_slice_accounting,
    )

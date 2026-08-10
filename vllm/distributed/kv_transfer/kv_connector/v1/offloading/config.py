# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Translate vLLM KV cache metadata for native offloading backends."""

from typing import TYPE_CHECKING

from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    FullAttentionSpec,
    KVCacheSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.kv_offload.config import (
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
    groups = tuple(
        OffloadingGroupConfig(
            tokens_per_block=(
                group.kv_cache_spec.block_size
                * (
                    parallel_config.decode_context_parallel_size
                    if isinstance(group.kv_cache_spec, AttentionSpec)
                    else 1
                )
            ),
            layer_names=tuple(group.layer_names),
        )
        for group in kv_cache_config.kv_cache_groups
    )

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

    single_group_spec = (
        kv_cache_config.kv_cache_groups[0].kv_cache_spec
        if len(kv_cache_config.kv_cache_groups) == 1
        else None
    )
    replicated_layout = (
        vllm_config.model_config.use_mla
        # Exact type: fail closed on wrappers and sliding-window variants.
        and type(single_group_spec) is MLAAttentionSpec
        # Page accounting: one MLA page per layer, no packed/mixed rows.
        and worker_kv_bytes_per_block > 0
        and worker_kv_bytes_per_block
        == single_group_spec.page_size_bytes
        * len(kv_cache_config.kv_cache_groups[0].layer_names)
        # Safe MVP boundary: TP-only, no other parallel axes.
        and parallel_config.tensor_parallel_size > 1
        and parallel_config.pipeline_parallel_size == 1
        and parallel_config.prefill_context_parallel_size == 1
        and parallel_config.decode_context_parallel_size == 1
        and parallel_config.world_size == parallel_config.tensor_parallel_size
        # Shared /dev/shm mmap layout is single-node mp only.
        and parallel_config.distributed_executor_backend == "mp"
        and parallel_config.nnodes_within_dp == 1
    )

    canonical_layout = bool(extra_config.get("canonical_layout", False))
    canonical_format = None
    if canonical_layout:
        from vllm.config import set_current_vllm_config

        from .canonical_mapping import canonical_format_id

        # canonical_format_id resolves the KV cache layout, which needs the
        # current vLLM config; scheduler-side consumers run outside that
        # context, so resolve the id here once.
        with set_current_vllm_config(vllm_config):
            canonical_format = canonical_format_id()

    # Only a single non-MLA full-attention group with genuinely head-sharded
    # pages is parallelism-invariant: replicated latent or GQA heads,
    # per-token-head scales, CP token sharding, and the V2 model runner's
    # layout are all excluded.
    is_parallelism_agnostic = (
        not vllm_config.use_v2_model_runner
        and single_group_spec is not None
        and isinstance(single_group_spec, FullAttentionSpec)
        and not isinstance(single_group_spec, MLAAttentionSpec)
        and single_group_spec.num_kv_heads * parallel_config.tensor_parallel_size
        == vllm_config.model_config.get_total_num_kv_heads()
        and not single_group_spec.kv_quant_mode.is_per_token_head
        and parallel_config.decode_context_parallel_size == 1
        and parallel_config.prefill_context_parallel_size == 1
    )
    # Canonical pages are topology-free by construction, so the canonical
    # layout widens the gate to every config whose mappings derive portable,
    # group by group: sharded or replicated GQA heads, the TP-replicated MLA
    # latent, and attention-only hybrids. Certification happens per layer
    # against live tensor strides at registration, and create_worker fails
    # closed against this flag if any layer cannot be certified.
    if canonical_layout and not is_parallelism_agnostic:
        tp_size = parallel_config.tensor_parallel_size
        total_kv_heads = vllm_config.model_config.get_total_num_kv_heads()

        def spec_certifiable(spec: KVCacheSpec) -> bool:
            """Statically mirrors _layer_mapping's certifiable spec classes;
            hybrid attention models certify group by group."""
            if isinstance(spec, UniformTypeKVCacheSpecs):
                # Same-type layers with differing page sizes (e.g. MLA plus
                # its DSA indexer cache); mappings derive per inner spec
                return len(spec.kv_cache_specs) > 0 and all(
                    spec_certifiable(inner) for inner in spec.kv_cache_specs.values()
                )
            if not isinstance(spec, AttentionSpec):
                return False
            if spec.kv_quant_mode.is_per_token_head:
                return False
            if type(spec) is MLAAttentionSpec:
                return (
                    spec.compress_ratio == 1
                    and spec.real_page_size_bytes % spec.block_size == 0
                )
            if isinstance(spec, (SlidingWindowMLASpec, MLAAttentionSpec)):
                return False
            if not isinstance(spec, (FullAttentionSpec, SlidingWindowSpec)):
                return False
            return total_kv_heads % tp_size == 0 or tp_size % total_kv_heads == 0

        is_parallelism_agnostic = (
            len(kv_cache_config.kv_cache_groups) > 0
            and all(
                spec_certifiable(group.kv_cache_spec)
                for group in kv_cache_config.kv_cache_groups
            )
            and parallel_config.decode_context_parallel_size == 1
            and parallel_config.prefill_context_parallel_size == 1
            and parallel_config.world_size == tp_size
        )

    kv_events_config = vllm_config.kv_events_config
    cache_dtype = (
        vllm_config.model_config.dtype
        if vllm_config.cache_config.cache_dtype == "auto"
        else vllm_config.cache_config.cache_dtype
    )

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
            dtype=str(cache_dtype).removeprefix("torch."),
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
        replicated_layout=replicated_layout,
        canonical_layout=canonical_layout,
        canonical_format=canonical_format,
    )

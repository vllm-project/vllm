# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Node-local DP discovery and shared CPU allocation for simple offloading."""

import os
import uuid
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.distributed.parallel_state import (
    GroupCoordinator,
    get_dp_group,
    in_the_same_node_as,
)
from vllm.logger import init_logger
from vllm.utils.hashing import get_hash_fn_by_name
from vllm.utils.math_utils import round_up
from vllm.v1.core.kv_cache_utils import resolve_none_hash_seed
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion
from vllm.v1.simple_kv_offload.metadata import SimpleCPUOffloadHandshake

logger = init_logger(__name__)


def shared_hash_signature(config: VllmConfig) -> tuple[str, bytes, bytes]:
    """Check both the root hash and a chained block, including serialization."""
    algorithm = config.cache_config.prefix_caching_hash_algo
    seed = os.getenv("PYTHONHASHSEED")
    if seed == "random" or (algorithm.startswith("xxhash") and seed is None):
        raise ValueError(
            "cpu_offload_shared requires reproducible prefix hashes: use sha256 "
            "or sha256_cbor, or set the same numeric PYTHONHASHSEED on every DP rank."
        )
    hash_fn = get_hash_fn_by_name(algorithm)
    root = hash_fn(resolve_none_hash_seed(hash_fn))
    return algorithm, root, hash_fn((root, (1, 2, 3), None))


def validate_shared_config(
    config: VllmConfig, backend: str, lazy_offload: bool = False
) -> None:
    parallel = config.parallel_config
    if backend != "cpu":
        raise ValueError("cpu_offload_shared requires kv_offload_backend='cpu'.")
    if lazy_offload:
        raise ValueError("cpu_offload_shared currently requires eager offloading.")
    if not config.cache_config.enable_prefix_caching:
        raise ValueError("cpu_offload_shared requires prefix caching.")
    if (
        parallel.pipeline_parallel_size != 1
        or parallel.decode_context_parallel_size != 1
        or parallel.prefill_context_parallel_size != 1
    ):
        raise ValueError("cpu_offload_shared currently requires PP=DCP=PCP=1.")
    if parallel.enable_elastic_ep:
        raise ValueError("cpu_offload_shared requires a fixed DP topology.")
    # Adapter IDs are assigned independently by each engine and are part of
    # prefix identity. They are insufficient to certify shared adapter weights.
    if config.lora_config is not None:
        raise ValueError(
            "cpu_offload_shared does not support independently loaded LoRA."
        )
    shared_hash_signature(config)


def _discover_local_ranks(group: GroupCoordinator) -> list[int]:
    """Every rank participates in each node's collective IPC probe."""
    remaining = set(range(group.world_size))
    local_ranks: list[int] = []
    while remaining:
        first = min(remaining)
        peers = [
            rank
            for rank, local in enumerate(in_the_same_node_as(group.cpu_group, first))
            if local
        ]
        if not peers:
            raise RuntimeError("Could not discover shared offload IPC peers.")
        if group.rank_in_group in peers:
            local_ranks = peers
        remaining.difference_update(peers)
    return local_ranks


def allocate_shared_offload(
    config: VllmConfig,
    kv_cache_config: KVCacheConfig,
    gpu_caches: dict[str, torch.Tensor],
    blocks_per_rank: int,
) -> tuple[SharedOffloadRegion, dict[str, torch.Tensor], SimpleCPUOffloadHandshake]:
    """Allocate one layer-major mmap shared by the DP workers on each node."""
    group = get_dp_group()
    layout = tuple(
        (name, tuple(t.shape[1:]), str(t.dtype)) for name, t in gpu_caches.items()
    )
    signature = (
        config.model_config.compute_hash(),
        config.cache_config.compute_hash(),
        shared_hash_signature(config),
        repr(kv_cache_config.kv_cache_groups),
        blocks_per_rank,
        layout,
    )
    signatures: list[Any] = [None] * group.world_size
    torch.distributed.all_gather_object(signatures, signature, group=group.cpu_group)
    if any(peer != signature for peer in signatures):
        raise ValueError(
            "cpu_offload_shared DP hash/model/KV layout/capacity mismatch; "
            "all DP ranks must use identical cache identity and layout."
        )

    # Probe actual shared-memory visibility, including IPC namespaces. The
    # local_world_size used by OffloadingConnector describes workers inside one
    # engine; this group contains the DP replicas that share this pool.
    local_ranks = _discover_local_ranks(group)
    local_rank = local_ranks.index(group.rank_in_group)
    local_size = len(local_ranks)
    session = group.broadcast_object(
        uuid.uuid4().hex if group.rank_in_group == 0 else None
    )
    region_id = f"simple_{session}_{local_ranks[0]}"
    num_blocks = blocks_per_rank * local_size
    total_bytes = sum(t[0].nbytes for t in gpu_caches.values()) * num_blocks
    # A single chunk contains the entire pool, with one contiguous array per
    # layer. Keeping each layer's block stride matches the existing DMA backend.
    region = SharedOffloadRegion(
        engine_id=region_id,
        num_chunks=1,
        rank=None,
        kv_bytes_per_chunk=round_up(
            total_bytes, SharedOffloadRegion.BLOCK_SIZE_ALIGNMENT
        ),
        cpu_page_size=0,
        barrier=group.barrier,
        populate_only_on_creator=True,
    )
    cpu_caches = {
        name: region.create_next_canonical_view(num_blocks * tensor[0].nbytes)
        .view(tensor.dtype)
        .view((num_blocks,) + tensor.shape[1:])
        for name, tensor in gpu_caches.items()
    }
    metadata = SimpleCPUOffloadHandshake(
        region_id=region_id,
        local_rank=local_rank,
        local_size=local_size,
        blocks_per_rank=blocks_per_rank,
        hash_signature=signature[2],
    )
    logger.info(
        "SimpleCPU shared offload: DP rank %d, local peers=%s, "
        "%d slots per rank, %.2f GiB per node. Sharing is node-local.",
        config.parallel_config.data_parallel_rank,
        local_ranks,
        blocks_per_rank,
        region.total_size_bytes / 1024**3,
    )
    return region, cpu_caches, metadata

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared types for the ECCPUConnector scheduler and worker delegates."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorMetadata
from vllm.distributed.ec_transfer.ec_connector.cpu.ec_shared_region import (
    ECSharedRegion,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig


@dataclass
class ECCPUConnectorMetadata(ECConnectorMetadata):
    """Per-step scheduler → worker payload for the ECCPUConnector.

    Populated by `ECCPUScheduler.build_connector_meta`; consumed by
    `ECCPUWorker` via the mixin's `bind_connector_metadata`.
    """

    # Producer role: mm_hashes the scheduler has just allocated CPU
    # blocks for this step; the worker's save_caches copies
    # encoder_cache[mm_hash] → mmap at these block IDs.
    saves: dict[str, list[int]] = field(default_factory=dict)

    # Consumer role: mm_hashes whose bytes are available in the local mmap;
    # the worker's start_load_caches copies mmap[block_ids] → GPU
    # encoder_cache.
    loads: dict[str, list[int]] = field(default_factory=dict)


def _get_encoder_cache_hidden_dim(vllm_config: "VllmConfig") -> int:
    """Return the per-token hidden dimension for encoder cache entries.

    For most models this equals the LLM's hidden size.  Qwen3-VL (and any
    future model with deepstack visual encoding) is an exception: the ViT
    concatenates its own output with features from N decoder layers before
    storing in encoder_cache, producing a tensor of width
    ``out_hidden_size * (1 + N)`` per visual token.  Using the plain LLM
    hidden size would under-allocate EC blocks and silently truncate the
    transferred data, leading to a shape mismatch on the consumer.
    """
    model_config = vllm_config.model_config
    hf_config = getattr(model_config, "hf_config", None)
    vision_config = (
        getattr(hf_config, "vision_config", None) if hf_config is not None else None
    )
    if vision_config is not None:
        out_hidden_size = getattr(vision_config, "out_hidden_size", None)
        deepstack_indexes = getattr(vision_config, "deepstack_visual_indexes", None)
        if out_hidden_size is not None and deepstack_indexes:
            return out_hidden_size * (1 + len(deepstack_indexes))
    return model_config.get_inputs_embeds_size()


def create_ec_shared_region(vllm_config: "VllmConfig") -> ECSharedRegion:
    """Build the EC mmap region from `vllm_config`.

    Both `ECCPUScheduler` and `ECCPUWorker` call this to get the same
    shared region (same engine_id, same block_size_bytes).
    """
    ec_config = vllm_config.ec_transfer_config
    assert ec_config is not None, "ec_transfer_config required to build region"

    dp_rank = vllm_config.parallel_config.data_parallel_rank
    engine_id = f"{vllm_config.instance_id}_dp{dp_rank}"

    dtype = vllm_config.model_config.dtype
    hidden_dim = _get_encoder_cache_hidden_dim(vllm_config)
    element_size = torch.empty(0, dtype=dtype).element_size()
    block_size_bytes = hidden_dim * element_size

    cpu_bytes = ec_config.ec_connector_extra_config.get("ec_cpu_bytes")
    if not cpu_bytes:
        raise ValueError("ec_cpu_bytes must be specified in ec_connector_extra_config")
    cpu_bytes = int(cpu_bytes)
    num_blocks = cpu_bytes // block_size_bytes

    return ECSharedRegion(
        engine_id=engine_id,
        num_blocks=num_blocks,
        block_size_bytes=block_size_bytes,
    )

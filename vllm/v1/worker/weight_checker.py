# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-worker weight checksums and the reset that supports verifying them."""

import hashlib

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.distributed.parallel_state import (
    get_ep_group,
    get_pcp_group,
    get_pp_group,
    get_tp_group,
)
from vllm.v1.worker.utils import _iter_checksum_targets, _randomize_tensor_inplace


def _rank_prefix(vllm_config: VllmConfig, dp_rank: int) -> str:
    """Return the rank-qualified key prefix for this worker's tensors."""
    pcp_rank = get_pcp_group().rank_in_group
    pp_rank = get_pp_group().rank_in_group
    tp_rank = get_tp_group().rank_in_group
    ep_rank = get_ep_group().rank_in_group if vllm_config.model_config.is_moe else 0
    return f"dp{dp_rank}:pp{pp_rank}:pcp{pcp_rank}:tp{tp_rank}:ep{ep_rank}:"


def compute_weight_checksums(
    model: nn.Module, vllm_config: VllmConfig, dp_rank: int
) -> dict[str, str]:
    """Return one SHA-256 digest per checksum-covered tensor on this worker.

    Hashing needs host bytes, so each tensor is moved to CPU as one uint8
    array and passed to hashlib as a buffer.
    """
    prefix = _rank_prefix(vllm_config, dp_rank)
    checksums: dict[str, str] = {}
    for name, tensor in _iter_checksum_targets(model):
        cpu_uint8 = tensor.data.contiguous().cpu().view(torch.uint8).numpy()
        # Hash the array in place; .tobytes() would copy it a second time.
        raw = memoryview(cpu_uint8)
        checksums[f"{prefix}{name}"] = hashlib.sha256(raw).hexdigest()
    return checksums


def reset_weights(model: nn.Module) -> None:
    """Randomize exactly the tensors covered by ``compute_weight_checksums``."""
    for _, tensor in _iter_checksum_targets(model):
        # Chunk so the staging buffer stays bounded for large weights.
        if tensor.numel() == 0:
            continue
        if tensor.is_contiguous():
            chunks = tensor.data.view(-1).split(64 * 1024 * 1024)
        elif tensor.ndim == 0:
            chunks = (tensor.data,)
        else:
            row_numel = tensor[0].numel()
            rows_per_chunk = max(1, (64 * 1024 * 1024) // row_numel)
            chunks = tensor.data.split(rows_per_chunk, dim=0)
        for chunk in chunks:
            _randomize_tensor_inplace(chunk)

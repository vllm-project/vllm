# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Helpers for EPLB expert-weight containers."""

from collections.abc import Iterator, Sequence
from typing import TypeAlias

import torch

from vllm.distributed.utils import is_weak_contiguous

EplbExpertWeight: TypeAlias = torch.Tensor | Sequence[torch.Tensor]
EplbLayerWeights: TypeAlias = Sequence[EplbExpertWeight]


def get_eplb_num_experts(weight: EplbExpertWeight) -> int:
    """Return the number of local experts represented by a weight."""
    if isinstance(weight, torch.Tensor):
        if weight.ndim == 0:
            raise ValueError(
                "Aggregated EPLB weights must have at least one dimension."
            )
        return weight.shape[0]
    return len(weight)


def get_eplb_weight_device(weight: EplbExpertWeight) -> torch.device:
    """Return the device shared by all tensors in an EPLB weight."""
    if isinstance(weight, torch.Tensor):
        return weight.device
    if not weight:
        raise ValueError("Per-expert EPLB weight sequences must not be empty.")
    return weight[0].device


def get_eplb_expert_tensor(
    weight: EplbExpertWeight,
    expert_id: int,
) -> torch.Tensor:
    """Return one expert tensor from an aggregated or per-expert weight."""
    return weight[expert_id]


def empty_eplb_weight_like(weight: EplbExpertWeight) -> EplbExpertWeight:
    """Allocate an independent EPLB buffer with the same container structure."""
    if isinstance(weight, torch.Tensor):
        return torch.empty_like(weight)
    buffers = [torch.empty_like(tensor) for tensor in weight]
    return tuple(buffers) if isinstance(weight, tuple) else buffers


def validate_eplb_weight(
    weight: EplbExpertWeight,
    local_num_experts: int,
) -> None:
    """Validate an aggregated or per-expert EPLB weight container."""
    if not isinstance(weight, torch.Tensor) and not weight:
        raise ValueError("Per-expert EPLB weight sequences must not be empty.")

    num_experts = get_eplb_num_experts(weight)
    if num_experts != local_num_experts:
        raise ValueError(
            "EPLB weight has an unexpected number of local experts: "
            f"expected={local_num_experts}, got={num_experts}."
        )

    if isinstance(weight, torch.Tensor):
        return

    if not all(isinstance(tensor, torch.Tensor) for tensor in weight):
        raise TypeError("Every per-expert EPLB weight must be a torch.Tensor.")

    first = weight[0]
    for expert_id, tensor in enumerate(weight[1:], start=1):
        if tensor.device != first.device:
            raise ValueError(
                "Per-expert EPLB weights must use one device: "
                f"expert 0 is on {first.device}, expert {expert_id} is on "
                f"{tensor.device}."
            )
        if tensor.dtype != first.dtype:
            raise ValueError(
                "Per-expert EPLB weights must use one dtype: "
                f"expert 0 uses {first.dtype}, expert {expert_id} uses "
                f"{tensor.dtype}."
            )
        if tensor.shape != first.shape:
            raise ValueError(
                "Per-expert EPLB weights must have one shape: "
                f"expert 0 has {tuple(first.shape)}, expert {expert_id} has "
                f"{tuple(tensor.shape)}."
            )
        if tensor.layout != first.layout or tensor.stride() != first.stride():
            raise ValueError(
                "Per-expert EPLB weights must have matching layouts and strides."
            )

    if not all(is_weak_contiguous(tensor) for tensor in weight):
        raise ValueError("Per-expert EPLB weights must be contiguous in memory.")

    storage_ids = [
        storage.data_ptr()
        for tensor in weight
        if (storage := tensor.untyped_storage()).nbytes() > 0
    ]
    if len(set(storage_ids)) != len(storage_ids):
        raise ValueError("Per-expert EPLB weights must use independent tensor storage.")


def is_eplb_weight_aggregated(weight: EplbExpertWeight) -> bool:
    """Return whether a weight uses the aggregated tensor representation."""
    return isinstance(weight, torch.Tensor)


def iter_eplb_weight_tensors(
    weight: EplbExpertWeight,
) -> Iterator[torch.Tensor]:
    """Iterate over the tensors owned by one EPLB weight container."""
    if isinstance(weight, torch.Tensor):
        yield weight
    else:
        yield from weight

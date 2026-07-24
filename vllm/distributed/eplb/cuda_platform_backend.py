# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA-compatible EPLB Platform Backend."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

import torch

from vllm.triton_utils import tl, triton

from .platform_backend import (
    EplbDeviceEvent,
    EplbDeviceRuntime,
    EplbPlatformBackend,
)

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

    from vllm.config import ParallelConfig
    from vllm.distributed.parallel_state import GroupCoordinator

    from .eplb_communicator import EplbCommunicator
    from .weight_utils import EplbExpertWeight, EplbLayerWeights


@triton.jit
def _eplb_map_and_record_i32_kernel(
    topk_ids_ptr,
    logical_replica_count_ptr,
    logical_to_physical_ptr,
    out_ids_ptr,
    out_ptr,
    record_enabled_ptr,
    num_unpadded_tokens_ptr,
    num_logical_experts,
    map_slots,
    out_size,
    numel,
    num_active_experts,
    HAS_NUM_UNPADDED: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel

    expert_id = tl.load(topk_ids_ptr + offs, mask=mask, other=0).to(tl.int64)
    valid_expert = (expert_id >= 0) & (expert_id < num_logical_experts)
    safe_expert_id = tl.where(valid_expert, expert_id, 0)

    # 1. Convert the logical expert ids to physical expert ids
    replica_count = tl.load(
        logical_replica_count_ptr + safe_expert_id,
        mask=mask & valid_expert,
        other=1,
    )
    # Avoid invalid modulo/div by forcing at least 1.
    replica_count = tl.maximum(replica_count, 1)
    # floor(2^32 / phi), classic Knuth multiplicative hash multiplier.
    KNUTH_MULTIPLIER = 2654435769
    token_idx = (offs // num_active_experts).to(tl.int64)
    hashed = (token_idx * KNUTH_MULTIPLIER) & 0xFFFFFFFF
    replica_idx = hashed % replica_count
    map_index = safe_expert_id * map_slots + replica_idx
    physical_id = tl.load(
        logical_to_physical_ptr + map_index,
        mask=mask & valid_expert,
        other=-1,
    )
    tl.store(out_ids_ptr + offs, physical_id, mask=mask)

    # 2. Record expert load metrics.

    # TODO(bowen): When using `FusedMoEModularKernel`, this
    # can be done in a more unified way, since
    # `FusedMoEPrepareAndFinalize` will return the expert
    # token count, in some cases directly from the kernel.
    # However, now there are many code paths not using
    # the modular kernel, e.g. calling `fused_experts`,
    # so we decide to keep the logic here.
    #
    # If later refactor moved all the MoE kernel calls
    # to the modular kernel, we can move this logic there
    # to achieve better efficiency.

    record_enabled = tl.load(record_enabled_ptr) != 0
    # Skip padded tokens when recording.
    if HAS_NUM_UNPADDED:
        num_unpadded_tokens = tl.load(num_unpadded_tokens_ptr)
        is_unpadded = offs < num_unpadded_tokens * num_active_experts
    else:
        is_unpadded = True
    valid = (
        mask
        & record_enabled
        & is_unpadded
        & (physical_id >= 0)
        & (physical_id < out_size)
    )
    safe_physical_id = tl.where(physical_id >= 0, physical_id, 0)
    tl.atomic_add(out_ptr + safe_physical_id, 1, mask=valid)


def eplb_map_to_physical_and_record(
    topk_ids: torch.Tensor,
    logical_to_physical_map: torch.Tensor,
    logical_replica_count: torch.Tensor,
    expert_load_view: torch.Tensor,
    record_enabled: torch.Tensor,
    num_unpadded_tokens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Map logical experts and optionally record their physical load."""
    # Fused triton implementation: mapping + optional recording in one kernel.
    topk_ids_in = topk_ids.contiguous().to(dtype=torch.int32)
    numel = topk_ids_in.numel()
    if numel == 0:
        return topk_ids
    num_active_experts = topk_ids_in.shape[-1]
    out_flat = torch.empty((numel,), device=topk_ids.device, dtype=topk_ids.dtype)
    grid = lambda meta: (triton.cdiv(numel, meta["BLOCK_SIZE"]),)
    assert expert_load_view.is_contiguous()
    _eplb_map_and_record_i32_kernel[grid](
        topk_ids_in,
        logical_replica_count.contiguous(),
        logical_to_physical_map.contiguous(),
        out_flat,
        expert_load_view,
        record_enabled,
        num_unpadded_tokens,
        logical_replica_count.shape[0],
        logical_to_physical_map.shape[1],
        expert_load_view.shape[0],
        numel,
        num_active_experts,
        HAS_NUM_UNPADDED=num_unpadded_tokens is not None,
        BLOCK_SIZE=256,
    )
    return out_flat.reshape(topk_ids.shape)


class CudaEplbDeviceRuntime(EplbDeviceRuntime):
    """CUDA-compatible stream and event operations."""

    def get_device_index(self, device: torch.device) -> int:
        if device.index is not None:
            return device.index
        return torch.accelerator.current_device_index()

    def set_device(self, device_index: int) -> None:
        torch.accelerator.set_device_index(device_index)

    def create_stream(self, device_index: int) -> Any:
        return torch.cuda.Stream(device=device_index)

    def stream_context(self, stream: Any) -> AbstractContextManager[Any]:
        return torch.cuda.stream(cast(torch.cuda.Stream, stream))

    def create_event(self, enable_timing: bool = False) -> EplbDeviceEvent:
        return cast(
            EplbDeviceEvent,
            torch.cuda.Event(enable_timing=enable_timing),
        )

    def synchronize(self, stream: Any | None = None) -> None:
        if stream is None:
            torch.accelerator.synchronize()
        else:
            stream.synchronize()


class CudaEplbPlatformBackend(EplbPlatformBackend):
    """Built-in EPLB operations shared by CUDA and ROCm."""

    def __init__(self) -> None:
        self._device_runtime = CudaEplbDeviceRuntime()

    @classmethod
    def resolve_communicator(cls, parallel_config: ParallelConfig) -> str:
        from .eplb_communicator import has_nixl

        # Prefer NIXL for zero-copy RDMA reads. Elastic EP otherwise needs
        # PyNccl, while static EP keeps the existing Gloo fallback.
        if has_nixl():
            return "nixl"
        if parallel_config.enable_elastic_ep:
            return "pynccl"
        return "torch_gloo"

    @classmethod
    def validate_config(cls, parallel_config: ParallelConfig) -> None:
        communicator = parallel_config.eplb_config.communicator
        assert communicator is not None
        use_async = parallel_config.eplb_config.use_async

        if communicator == "platform":
            raise ValueError(
                "The built-in CUDA/ROCm EPLB Backend does not provide a "
                "'platform' communicator."
            )
        if use_async and communicator in ("torch_nccl", "pynccl"):
            raise ValueError(
                f"{communicator} communicator is incompatible with async "
                "EPLB due to NCCL multi-stream conflicts. Use 'torch_gloo' "
                "or 'nixl' instead, or leave communicator unset for automatic "
                "selection."
            )

        if not parallel_config.enable_elastic_ep:
            return
        if communicator not in ("torch_nccl", "pynccl", "nixl"):
            raise ValueError(
                "Elastic EP requires the 'torch_nccl', 'pynccl', or 'nixl' "
                f"EPLB communicator (got {communicator!r})."
            )
        if use_async and communicator != "nixl":
            raise ValueError(
                "Elastic EP with async EPLB requires the NIXL communicator."
            )
        if communicator == "nixl":
            from .eplb_communicator import has_nixl

            if not has_nixl():
                raise ValueError(
                    "The NIXL package is required for the configured EPLB communicator."
                )

    def map_and_record(
        self,
        topk_ids: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
        expert_load_view: torch.Tensor,
        record_enabled: torch.Tensor,
        num_unpadded_tokens: torch.Tensor | None,
    ) -> torch.Tensor:
        return eplb_map_to_physical_and_record(
            topk_ids,
            logical_to_physical_map,
            logical_replica_count,
            expert_load_view,
            record_enabled,
            num_unpadded_tokens,
        )

    def create_communicator(
        self,
        group_coordinator: GroupCoordinator,
        expert_weights: Sequence[EplbLayerWeights],
        expert_buffer: Sequence[EplbExpertWeight],
    ) -> EplbCommunicator:
        raise ValueError(
            "The built-in CUDA/ROCm EPLB Backend does not provide a "
            "'platform' communicator."
        )

    @property
    def device_runtime(self) -> EplbDeviceRuntime:
        return self._device_runtime

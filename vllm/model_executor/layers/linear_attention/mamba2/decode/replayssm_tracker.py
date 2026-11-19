# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch

from vllm.v1.kv_cache_interface import MambaSpec

from .base import M, SSUBuffers
from .replayssm_base import MambaReplaySSMKernel

if TYPE_CHECKING:
    from vllm.config import VllmConfig


@dataclass
class TrackerBuffers(SSUBuffers):
    # Contiguous decode slots: full CUDA graphs read this address.
    state_indices_d: torch.Tensor


class TrackerReplaySSMKernel(MambaReplaySSMKernel):
    """A ReplaySSM kernel whose rings follow the trackers the model runner
    commits after each step, so accepted speculative tokens stay in the ring."""

    uses_ring_trackers = True

    @classmethod
    def allocate_buffers(
        cls,
        vllm_config: "VllmConfig",
        kv_cache_spec: MambaSpec,
        max_cudagraph_decodes: int,
        device: torch.device,
    ) -> TrackerBuffers:
        return TrackerBuffers(
            torch.empty((max_cudagraph_decodes,), dtype=torch.int32, device=device)
        )

    @classmethod
    def prepare_metadata(
        cls, buffers: SSUBuffers | None, metadata: M, num_rows: int, padded: bool
    ) -> M:
        assert isinstance(buffers, TrackerBuffers)
        assert metadata.state_indices_tensor_d is not None
        slots = metadata.state_indices_tensor_d[:, 0]
        if padded:
            buffers.state_indices_d[:num_rows].copy_(slots, non_blocking=True)
            indices = buffers.state_indices_d[:num_rows]
        else:
            indices = slots.contiguous()
        return replace(metadata, replayssm_state_indices_d=indices)

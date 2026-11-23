# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.config import VllmConfig
from vllm.model_executor.layers.mamba.checkpoint import (
    MambaPrefillCheckpointExporter,
    MambaPrefillCheckpointMetadata,
)
from vllm.third_party.flash_linear_attention.ops.utils import FLA_CHUNK_SIZE
from vllm.v1.worker.workspace import (
    current_workspace_manager,
    is_workspace_manager_initialized,
)

TritonKDACheckpointSpec = tuple[tuple[int, ...], torch.dtype]

# The Triton chunk kernel sees the recurrent state only at chunk boundaries.
_KDA_PREFILL_CHECKPOINT_ALIGNMENT = {"flashkda": 16, "triton": FLA_CHUNK_SIZE}


def kda_prefill_checkpoint_alignment(backend: str) -> int | None:
    return _KDA_PREFILL_CHECKPOINT_ALIGNMENT.get(backend)


def reserve_triton_kda_checkpoint_buffer(
    vllm_config: VllmConfig, num_heads: int, head_dim: int, dtype: torch.dtype
) -> TritonKDACheckpointSpec:
    """Reserve the `[max_num_seqs, H, V, K]` workspace buffer the Triton chunk
    kernel stores checkpoint states into; profiling skips attention, so the
    layer reserves it before the KV cache is sized."""
    max_seqs = vllm_config.scheduler_config.max_num_seqs
    spec = ((max_seqs, num_heads, head_dim, head_dim), dtype)
    if is_workspace_manager_initialized():
        current_workspace_manager().get_simultaneous(spec)
    return spec


def get_triton_kda_checkpoint_buffer(
    checkpoint: MambaPrefillCheckpointMetadata | None,
    spec: TritonKDACheckpointSpec | None,
    num_seqs: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """The `checkpoint_state` and `checkpoint_offsets` to pass the Triton chunk
    kernel, or `None`s when the batch has no checkpoint."""
    if checkpoint is None:
        return None, None
    assert spec is not None
    (checkpoint_state,) = current_workspace_manager().get_simultaneous(spec)
    return checkpoint_state[:num_seqs], checkpoint.checkpoint_offsets


class FlashKDAPrefillCheckpointExporter(MambaPrefillCheckpointExporter):
    """Store KDA (FlashKDA or Triton) recurrent and convolution checkpoint states."""

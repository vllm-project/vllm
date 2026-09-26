# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Conv half of the ROCm Kimi-K3 prefill checkpoint."""

from dataclasses import dataclass

import torch

from vllm.model_executor.layers.mamba.checkpoint import (
    MambaPrefillCheckpointMetadata,
)
from vllm.model_executor.layers.mamba.kda_checkpoint import (
    FlashKDAPrefillCheckpointExporter,
)


@dataclass(frozen=True)
class KimiK3ROCmKDAPrefillCheckpointExporter(FlashKDAPrefillCheckpointExporter):
    """Conv window only; `fused_kda_chunk` writes the recurrent half itself.

    `state_len` is `conv_kernel_size - 1`, not the spec-widened conv row.
    """

    def export(  # type: ignore[override]
        self,
        checkpoint: MambaPrefillCheckpointMetadata,
        *,
        raw_qkv: torch.Tensor,
        conv_state: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> None:
        assert self.state_len is not None
        # An empty recurrent checkpoint masks the kernel's recurrent half off.
        super().export(
            checkpoint,
            raw_qkv=raw_qkv,
            conv_state=conv_state,
            recurrent_checkpoint=conv_state[:, :0],
            recurrent_state=conv_state,
            cu_seqlens=cu_seqlens,
        )

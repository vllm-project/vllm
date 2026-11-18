# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Mamba2 SSD prefill scan's kernel interface."""

from dataclasses import replace
from typing import TYPE_CHECKING

import torch

from vllm.model_executor.layers.linear_attention.kernel import LinearAttnKernel
from vllm.model_executor.layers.mamba.ops.gather_initial_states import (
    gather_initial_states,
)
from vllm.model_executor.layers.mamba.ops.scatter_states import scatter_states
from vllm.v1.kv_cache_interface import MambaSpec

if TYPE_CHECKING:
    from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaMixer2
    from vllm.v1.attention.backends.mamba2_attn import Mamba2AttentionMetadata


class MambaSSDKernel(LinearAttnKernel):
    """The Mamba2 SSD prefill scan, including its state cache reads and writes."""

    @classmethod
    def prepare_metadata(
        cls,
        metadata: "Mamba2AttentionMetadata",
        kv_cache_spec: MambaSpec,
        num_computed_tokens_p_cpu: torch.Tensor,
        query_start_loc_p_cpu: torch.Tensor,
    ) -> "Mamba2AttentionMetadata":
        """Add the prefill tensors this kernel reads. The default is the chunk
        layout the Triton and CPU kernels read."""
        from vllm.v1.attention.backends.mamba_attn import (
            BaseMambaAttentionMetadataBuilder,
        )

        assert metadata.query_start_loc_p is not None
        checkpoint = metadata.checkpoint_meta
        # The host offsets place a chunk boundary on each checkpoint.
        cu_chunk_seqlen_p, seq_idx_p, last_chunk_indices_p, checkpoint_chunk_idx = (
            BaseMambaAttentionMetadataBuilder._build_chunk_metadata_tensors(
                metadata.chunk_size,
                num_computed_tokens_p_cpu,
                query_start_loc_p_cpu,
                metadata.query_start_loc_p.device,
                checkpoint.offsets if checkpoint is not None else None,
            )
        )
        return replace(
            metadata,
            cu_chunk_seqlen_p=cu_chunk_seqlen_p,
            seq_idx_p=seq_idx_p,
            last_chunk_indices_p=last_chunk_indices_p,
            checkpoint_chunk_idx=checkpoint_chunk_idx,
        )

    @classmethod
    def read_initial_states(
        cls, ssm_state: torch.Tensor, metadata: "Mamba2AttentionMetadata"
    ) -> torch.Tensor | None:
        """Each prefill's cached state, zeros where it has none."""
        if metadata.has_initial_states_p is None or not metadata.prep_initial_states:
            return metadata.zero_initial_states_p
        assert metadata.state_indices_tensor_p is not None
        return cls.gather_states(
            ssm_state, metadata.state_indices_tensor_p, metadata.has_initial_states_p
        )

    @classmethod
    def gather_states(
        cls,
        ssm_state: torch.Tensor,
        indices: torch.Tensor,
        has_initial_states: torch.Tensor,
    ) -> torch.Tensor:
        return gather_initial_states(ssm_state, indices, has_initial_states)

    @classmethod
    def write_final_states(
        cls,
        ssm_state: torch.Tensor,
        final_states: torch.Tensor,
        indices: torch.Tensor,
    ) -> None:
        scatter_states(ssm_state, final_states, indices)

    @classmethod
    def forward(
        cls,
        layer: "MambaMixer2",
        x: torch.Tensor,
        dt: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        out: torch.Tensor,
        ssm_state: torch.Tensor,
        metadata: "Mamba2AttentionMetadata",
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Scan the prefills from their cached states into `out`, and cache
        their final states.

        Returns:
            The checkpoint states and each prefill's row in them, or None when
            no checkpoints are exported.

        """
        assert metadata.state_indices_tensor_p is not None
        final_states, checkpoint_states = cls._scan(
            layer,
            x,
            dt,
            B,
            C,
            out,
            ssm_state,
            metadata,
            cls.read_initial_states(ssm_state, metadata),
        )
        if final_states is not None:
            cls.write_final_states(
                ssm_state, final_states, metadata.state_indices_tensor_p
            )
        if checkpoint_states is None:
            return None
        assert metadata.checkpoint_chunk_idx is not None
        return checkpoint_states, metadata.checkpoint_chunk_idx

    @classmethod
    def _scan(
        cls,
        layer: "MambaMixer2",
        x: torch.Tensor,
        dt: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        out: torch.Tensor,
        ssm_state: torch.Tensor,
        metadata: "Mamba2AttentionMetadata",
        initial_states: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Return the final states, or None if already written into
        `ssm_state`, and the checkpoint states, or None when none are
        exported."""
        raise NotImplementedError

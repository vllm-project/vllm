# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The causal conv1d prefill kernel interface."""

from typing import TYPE_CHECKING, TypeVar

import torch

from vllm.model_executor.layers.linear_attention.kernel import LinearAttnKernel

if TYPE_CHECKING:
    from vllm.v1.attention.backends.mamba_attn import BaseMambaAttentionMetadata

M = TypeVar("M", bound="BaseMambaAttentionMetadata")


class Conv1dPrefillKernel(LinearAttnKernel):
    """The causal conv1d over each prefill's tokens, starting from its cached
    conv state and writing back the state at its end."""

    @classmethod
    def register_warmup(
        cls,
        *,
        dim: int,
        state_len: int,
        dim_first: bool,
        x_row_stride: int,
        weight_dtype: torch.dtype,
        bias_dtype: torch.dtype | None,
        max_num_seqs: int,
        max_num_batched_tokens: int,
    ) -> None:
        """Register the JIT warmup of this kernel's launches for a layer of this
        geometry (per TP rank): `dim` channels, a conv state of `state_len`
        positions stored dim-first or not, and inputs whose rows are
        `x_row_stride` elements apart."""

    @classmethod
    def prepare_metadata(cls, metadata: M, query_start_loc_p_cpu: torch.Tensor) -> M:
        """Add the prefill tensors this kernel reads."""
        return metadata

    @classmethod
    def forward(
        cls,
        x: torch.Tensor,
        conv_states: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        activation: str | None,
        metadata: "BaseMambaAttentionMetadata",
    ) -> torch.Tensor:
        """Convolve the prefill tokens `x` (`(dim, num_tokens)`) and return the
        output in the same layout. Each prefill reads its initial state from,
        and writes its final state to, its slot in
        `metadata.state_indices_tensor_p` of `conv_states`
        (`(num_slots, dim, width - 1)`)."""
        raise NotImplementedError

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The specialized decode executor.

``SpecializedDecode`` runs the decoder layers of ``MiniMaxM3Model`` for FULL
CUDA graph captures of pure decode steps of 1-32 rows (``moe.MAX_ROWS``).
``plan`` decides per forward whether it runs; the backbone then hands its
layers to ``DecodeStep.run`` after the embedding and keeps its final norm.
Every other forward (prefill, mixed, eager, PIECEWISE, larger batches) runs
the stock layers, which the executor leaves unchanged.

Per layer the executor runs the stock modules through its own steps:

- the input norm, which finalizes the previous MoE layer in the fused
  ``ops/finalize_allreduce_norm.py`` kernel;
- the attention;
- the fused all-reduce + post-attention norm;
- the FFN: ``moe.LowBatchFfn`` on the MoE layers that leave their finalize to
  the next layer, the stock modules on the others.

The fused kernels issue TP collectives, so every decision reads host state that
all TP ranks share (the step's rows and the capture state); nothing is read
back from the GPU.
"""

from typing import TYPE_CHECKING, NamedTuple

import torch

from vllm.model_executor.layers.fused_allreduce_gemma_rms_norm import (
    fused_allreduce_gemma_rms_norm,
)
from vllm.model_executor.layers.fused_moe.moe_output import MoEOutput
from vllm.sequence import IntermediateTensors

from .. import captures_pure_decode
from ..moe import MAX_ROWS

if TYPE_CHECKING:
    from ...model import MiniMaxM3DecoderLayer, MiniMaxM3Model


class DecodeStep(NamedTuple):
    """One captured decode step."""

    rows: int
    executor: "SpecializedDecode"

    def run(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        aux_hidden_states: list[torch.Tensor],
    ) -> tuple[torch.Tensor | MoEOutput, torch.Tensor]:
        """The decoder layers on the embedding ``hidden_states``; returns what
        the last layer leaves for the final norm."""
        return self.executor.run(hidden_states, positions, aux_hidden_states)


class SpecializedDecode:
    """The decode executor over one backbone's local layers."""

    def __init__(self, backbone: "MiniMaxM3Model") -> None:
        self.backbone = backbone
        self.layers: list[MiniMaxM3DecoderLayer] = list(
            backbone.layers[backbone.start_layer : backbone.end_layer]
        )

    def plan(
        self,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
    ) -> DecodeStep | None:
        """The step this forward runs, or None to run the stock layers."""
        rows = positions.shape[0]
        if (
            intermediate_tensors is not None
            or not 0 < rows <= MAX_ROWS
            or not captures_pure_decode()
        ):
            return None
        return DecodeStep(rows, self)

    def run(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        aux_hidden_states: list[torch.Tensor],
    ) -> tuple[torch.Tensor | MoEOutput, torch.Tensor]:
        aux_layers = self.backbone.aux_hidden_state_layers
        residual = None
        for idx, layer in enumerate(self.layers, start=self.backbone.start_layer):
            hidden_states, residual = layer.forward_input_norm(hidden_states, residual)
            # Aux hidden state k is the residual stream entering layer k.
            if idx in aux_layers:
                aux_hidden_states.append(residual.clone())
            hidden_states = layer.self_attn(
                positions=positions, hidden_states=hidden_states
            )
            hidden_states, residual = fused_allreduce_gemma_rms_norm(
                hidden_states, residual, layer.post_attention_layernorm
            )
            hidden_states = self._ffn(layer, hidden_states)
        return hidden_states, residual

    @staticmethod
    def _ffn(
        layer: "MiniMaxM3DecoderLayer", hidden_states: torch.Tensor
    ) -> torch.Tensor | MoEOutput:
        if not layer.is_moe_layer:
            return layer.mlp(hidden_states)
        moe = layer.block_sparse_moe
        if not moe.defers_finalize(hidden_states.shape[0]):
            return moe(hidden_states)
        ffn = moe.low_batch_ffn
        if ffn is not None and ffn.accepts(hidden_states):
            return ffn.run(hidden_states)
        return moe.forward_unfinalized(hidden_states)

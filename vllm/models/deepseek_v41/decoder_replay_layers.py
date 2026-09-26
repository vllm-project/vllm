# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decoder-side SWA bounded replay: running the replay layers on their batch.

The layers past the last KV-source layer own nothing but sliding-window KV, so
in eager prefill steps they run on each request's last ``window`` rows only.
``DeepseekV41ModelState`` prepares those rows as a sub-batch with attention
metadata and a forward context of its own, like a microbatch;
``DecoderReplayLayers`` gathers the layer inputs by its rows, runs the layers
under that context and scatters the outputs back to batch rows. Steps that run
in a CUDA graph keep the layers on the whole batch, inside the graph.
"""

from collections.abc import Callable

import torch

from vllm.forward_context import ForwardContext, override_forward_context


class DecoderReplayLayers:
    """Runs the replay layers on the step's replay batch.

    ``run_layers`` takes a batch's layer inputs and returns its per-row
    outputs. ``row_buffers`` hold per-row results the source layer's indexer
    publishes for the layers after it; they are compacted to the replay rows
    in place.
    """

    def __init__(
        self,
        window: int,
        run_layers: Callable[..., tuple[torch.Tensor, ...]],
        row_buffers: list[torch.Tensor],
    ) -> None:
        self.window = window
        self.run_layers = run_layers
        self.row_buffers = row_buffers
        # The replay batch, set by the model state every step: its rows of the
        # batch and its forward context. None runs the layers on the batch.
        self.rows: torch.Tensor | None = None
        self.forward_context: ForwardContext | None = None

    def __call__(
        self, hidden_states: torch.Tensor, *states: torch.Tensor | None
    ) -> tuple[torch.Tensor, ...]:
        rows = self.rows
        if rows is None:
            return self.run_layers(hidden_states, *states)
        num_rows = rows.shape[0]
        for buf in self.row_buffers:
            buf[:num_rows].copy_(buf.index_select(0, rows))
        with override_forward_context(self.forward_context):
            row_outputs = self.run_layers(
                hidden_states.index_select(0, rows),
                *(None if t is None else t.index_select(0, rows) for t in states),
            )
        # The trimmed rows' outputs stay zero; nothing reads them.
        num_tokens = hidden_states.shape[0]
        return tuple(
            out.new_zeros((num_tokens, *out.shape[1:])).index_copy_(0, rows, out)
            for out in row_outputs
        )

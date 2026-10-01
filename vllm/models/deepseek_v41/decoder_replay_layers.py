# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decoder-side SWA bounded replay: running the replay layers on their batch.

The layers past the last KV-source layer own nothing but sliding-window KV, so
in eager prefill steps they run on each request's last ``window`` rows only.
``DeepseekV41ModelState`` prepares those rows as a sub-batch with attention
metadata and a forward context of its own, like a microbatch;
``DecoderReplayLayers`` gathers the layer inputs by its rows, runs the layers
under that context and scatters the outputs back to batch rows. Steps that run
in a CUDA graph keep the layers on the whole batch, inside the graph; an eager
step's replay batch may run in a graph of its own (decoder_replay_cudagraph.py).
"""

from collections.abc import Callable
from dataclasses import dataclass

import torch

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.forward_context import (
    ForwardContext,
    get_forward_context,
    in_piecewise_cudagraph,
    override_forward_context,
)
from vllm.model_executor.layers.fused_moe.moe_output import MoEOutput


@dataclass
class ReplayBatch:
    """The replay layers' batch of one forward: its rows of the batch, its
    forward context and, when it runs in a CUDA graph, the graph's runner."""

    rows: torch.Tensor
    forward_context: ForwardContext
    run_graph: Callable[..., tuple[torch.Tensor, ...]] | None = None


class DecoderReplayLayers:
    """Runs the replay layers on the step's replay batch.

    ``run_layers`` takes a batch's layer inputs and returns its per-row
    outputs. ``row_buffers`` hold per-row results the source layer's indexer
    publishes for the layers after it; they are compacted to the replay rows
    in place. ``metadata_prefixes`` are the attention metadata keys the layers
    read, so the replay batch builds only those.
    """

    def __init__(
        self,
        window: int,
        run_layers: Callable[..., tuple[torch.Tensor, ...]],
        row_buffers: list[torch.Tensor],
        metadata_prefixes: set[str],
    ) -> None:
        self.window = window
        self.run_layers = run_layers
        self.row_buffers = row_buffers
        self.metadata_prefixes = metadata_prefixes
        # Set by the model state every step; None runs the layers on the batch.
        self.replay_batch: ReplayBatch | None = None
        # PIECEWISE model graphs of at least this many tokens break out to run the
        # layers on the replay batch; set with the replay graphs.
        self.trim_threshold: int | None = None
        self._graph_outputs: list[torch.Tensor] | None = None

    def __call__(
        self, hidden_states: torch.Tensor | MoEOutput, *states: torch.Tensor | None
    ) -> tuple[torch.Tensor, ...]:
        if self.trim_threshold is not None and in_piecewise_cudagraph():
            batch_descriptor = get_forward_context().batch_descriptor
            assert batch_descriptor is not None
            if batch_descriptor.num_tokens >= self.trim_threshold:
                return self._run_in_graph_break(hidden_states, *states)
        return self._run(hidden_states, *states)

    @eager_break_during_capture
    def _run_in_graph_break(
        self, hidden_states: torch.Tensor | MoEOutput, *states: torch.Tensor | None
    ) -> tuple[torch.Tensor, ...]:
        """An eager break of the model's graph: the outputs land in fixed buffers,
        which the graph's next segment reads."""
        outputs = self._run(hidden_states, *states)
        if self._graph_outputs is None:
            # The largest model graph is captured first.
            self._graph_outputs = [torch.empty_like(out) for out in outputs]
        return tuple(
            buf[: out.shape[0]].copy_(out)
            for buf, out in zip(self._graph_outputs, outputs)
        )

    def _run(
        self, hidden_states: torch.Tensor | MoEOutput, *states: torch.Tensor | None
    ) -> tuple[torch.Tensor, ...]:
        replay_batch = self.replay_batch
        if replay_batch is None:
            return self.run_layers(hidden_states, *states)
        # A trimming step holds a prefill longer than the window, more tokens
        # than any step whose MoE leaves its finalize to the next layer.
        assert isinstance(hidden_states, torch.Tensor)
        rows = replay_batch.rows
        num_rows = rows.shape[0]
        for buf in self.row_buffers:
            buf[:num_rows].copy_(buf.index_select(0, rows))
        inputs = (hidden_states, *states)
        with override_forward_context(replay_batch.forward_context):
            if replay_batch.run_graph is not None:
                row_outputs = replay_batch.run_graph(rows, inputs)
            else:
                row_outputs = self.run_layers(
                    *(None if t is None else t.index_select(0, rows) for t in inputs)
                )
        # The trimmed rows' outputs stay zero; nothing reads them.
        num_tokens = hidden_states.shape[0]
        return tuple(
            out.new_zeros((num_tokens, *out.shape[1:])).index_copy_(0, rows, out)
            for out in row_outputs
        )

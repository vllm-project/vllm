# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decoder-side SWA bounded replay: running the replay layers on their batch.

The layers past the last KV-source layer own nothing but sliding-window KV, and
the rest of the last KV source feeds only them; so once that layer has written
its KV for every row, it and the layers past it run on each request's last
``window`` rows only.
``DeepseekV41ModelState`` prepares those rows as a sub-batch with attention
metadata and a forward context of its own, like a microbatch;
``DecoderReplayLayers`` gathers the layer inputs by its rows, runs the layers
under that context and scatters the outputs back to batch rows. FULL graphs and
small PIECEWISE ones keep the layers on the whole batch; larger PIECEWISE graphs
break out to the replay batch, which may run in a graph of its own
(decoder_replay_cudagraph.py).
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
    """One forward's replay batch; ``run_graph`` is set when it runs in a graph."""

    rows: torch.Tensor
    forward_context: ForwardContext
    run_graph: Callable[..., tuple[torch.Tensor, ...]] | None = None


class DecoderReplayLayers:
    """Runs the replay layers on the step's replay batch.

    ``run_layers`` takes a batch's layer inputs and returns its per-row
    outputs. ``metadata_prefixes`` are the attention metadata keys the layers
    read, so the replay batch builds only those. ``first_swa_prefix`` is the
    first replay layer's sliding-window key: that layer wrote every row's window
    KV, so its metadata keeps the batch's window starts.
    """

    def __init__(
        self,
        window: int,
        run_layers: Callable[..., tuple[torch.Tensor, ...]],
        metadata_prefixes: set[str],
        first_swa_prefix: str,
    ) -> None:
        self.window = window
        self.run_layers = run_layers
        self.metadata_prefixes = metadata_prefixes
        self.first_swa_prefix = first_swa_prefix
        # Set by the model state every step; None runs the layers on the batch.
        self.replay_batch: ReplayBatch | None = None
        # Set by the model state when the replay layers run in graphs: PIECEWISE
        # model graphs of this many tokens or more break out to the replay batch.
        self.trim_threshold: int | None = None
        self._break_outputs: list[torch.Tensor] | None = None

    def __call__(
        self, hidden_states: torch.Tensor | MoEOutput, *states: torch.Tensor | None
    ) -> tuple[torch.Tensor, ...]:
        if in_piecewise_cudagraph() and self.replays():
            return self._run_in_graph_break(hidden_states, *states)
        return self._run(hidden_states, *states)

    def replays(self) -> bool:
        """Whether this forward runs the replay layers on a replay batch. In a
        PIECEWISE graph that is fixed when the graph is captured, so it follows
        the graph's size (``graph_replays``); the batch itself is set only at
        runtime. Otherwise a replay batch is set or not."""
        if in_piecewise_cudagraph():
            batch_descriptor = get_forward_context().batch_descriptor
            assert batch_descriptor is not None
            return self.graph_replays(batch_descriptor.num_tokens)
        return self.replay_batch is not None

    def graph_replays(self, num_tokens: int) -> bool:
        """Whether a PIECEWISE model graph of ``num_tokens`` padded tokens breaks
        out to the replay batch: the one policy its capture and the model state's
        runtime replay batch follow."""
        return self.trim_threshold is not None and num_tokens >= self.trim_threshold

    @eager_break_during_capture
    def _run_in_graph_break(
        self, hidden_states: torch.Tensor | MoEOutput, *states: torch.Tensor | None
    ) -> tuple[torch.Tensor, ...]:
        """A graph break: the outputs land in fixed buffers the next segment reads."""
        outputs = self._run(hidden_states, *states)
        if self._break_outputs is None:
            # Sized by the first call: the largest model graph is captured first.
            self._break_outputs = [torch.empty_like(out) for out in outputs]
        return tuple(
            buf[: out.shape[0]].copy_(out)
            for buf, out in zip(self._break_outputs, outputs)
        )

    def _run(
        self, hidden_states: torch.Tensor | MoEOutput, *states: torch.Tensor | None
    ) -> tuple[torch.Tensor, ...]:
        replay_batch = self.replay_batch
        if replay_batch is None:
            return self.run_layers(hidden_states, *states)
        # Replay batch steps hold more than `window` tokens: no deferred MoE finalize.
        assert isinstance(hidden_states, torch.Tensor)
        rows = replay_batch.rows
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

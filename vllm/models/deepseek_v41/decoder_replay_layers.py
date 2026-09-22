# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decoder-side SWA bounded replay: running the replay layers on their batch.

The layers past the last KV-source layer own nothing but sliding-window KV, so
in prefill they run on each request's last ``window`` rows only.
``DeepseekV41ModelState`` prepares those rows as a sub-batch with attention
metadata and a forward context of its own, like a microbatch;
``DecoderReplayLayers`` gathers the layer inputs by its rows, runs the layers
under that context and scatters the outputs back to batch rows.

The replay layers always run eagerly. Under a piecewise CUDA graph they are an
eager break of the model graph, captured once and run on every replay of it,
so like attention they write into outputs the caller allocates in the graph.
"""

from collections.abc import Callable

import torch

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import CUDAGraphMode
from vllm.forward_context import (
    ForwardContext,
    get_forward_context,
    override_forward_context,
)


class DecoderReplayLayers:
    """Runs the replay layers on the step's replay batch.

    ``run_layers`` takes a batch's layer inputs and returns its outputs, and
    ``new_outputs`` takes the same inputs and returns zeroed outputs of those
    shapes. ``row_buffers`` hold per-row results the source layer's indexer
    publishes for the layers after it; they are compacted to the replay rows
    in place.
    """

    def __init__(
        self,
        window: int,
        run_layers: Callable[..., tuple[torch.Tensor, ...]],
        new_outputs: Callable[..., tuple[torch.Tensor, ...]],
        row_buffers: list[torch.Tensor],
    ) -> None:
        self.window = window
        self.run_layers = run_layers
        self.new_outputs = new_outputs
        self.row_buffers = row_buffers
        # The replay batch, set by the model state every step: its rows of the
        # batch and its forward context. None runs the layers on the batch.
        self.rows: torch.Tensor | None = None
        self.forward_context: ForwardContext | None = None
        # Wrapped here rather than at import so the breakable-graph flag is
        # read after the config has set it.
        self._eager_break = eager_break_during_capture(self._run)

    def __call__(self, *states: torch.Tensor | None) -> tuple[torch.Tensor, ...]:
        in_piecewise_graph = (
            get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE
        )
        if self.rows is None and not in_piecewise_graph:
            return self.run_layers(*states)
        # A piecewise graph takes the eager break even when nothing trims: it is
        # captured on such a batch and replayed on the batches that do.
        outputs = self.new_outputs(*states)
        self._eager_break(outputs, *states)
        return outputs

    def _run(
        self, outputs: tuple[torch.Tensor, ...], *states: torch.Tensor | None
    ) -> None:
        """Run the layers into ``outputs``, on the replay rows when there are
        any. The trimmed rows' outputs stay zero; nothing reads them."""
        rows = self.rows
        if rows is None:
            torch._foreach_copy_(outputs, self.run_layers(*states))
            return
        num_rows = rows.shape[0]
        for buf in self.row_buffers:
            buf[:num_rows].copy_(buf.index_select(0, rows))
        with override_forward_context(self.forward_context):
            row_outputs = self.run_layers(
                *(None if t is None else t.index_select(0, rows) for t in states)
            )
        for out, row_out in zip(outputs, row_outputs):
            out.index_copy_(0, rows, row_out)

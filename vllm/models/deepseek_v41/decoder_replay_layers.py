# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decoder-side SWA bounded replay: running the replay layers on their batch.

Once the last KV-source layer has written every row's KV, it and the layers past
it (which own only SWA KV) run on each request's last ``window`` rows only.
``DeepseekV41ModelState`` prepares those rows as a sub-batch with attention
metadata and a forward context of its own, like a microbatch;
``DecoderReplayLayers`` gathers the layer inputs by its rows, runs the layers
under that context and scatters the outputs back to batch rows. FULL graphs and
small PIECEWISE ones keep the layers on the whole batch; larger PIECEWISE graphs
break out to the replay batch, which may run in a graph of its own
(decoder_replay_cudagraph.py).
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.forward_context import (
    ForwardContext,
    get_forward_context,
    in_piecewise_cudagraph,
    override_forward_context,
)
from vllm.model_executor.layers.fused_moe.moe_output import MoEOutput


def exact_replay_rows(window: int, num_layers: int) -> int:
    """Trailing rows to recompute so that the last ``window`` rows' outputs come
    out as in the full forward after ``num_layers`` sliding-window layers: a
    row's window reaches ``window - 1`` rows back, so each layer the exact rows
    lose that many at their low end."""
    return window + num_layers * (window - 1)


def exact_replay_stages(
    window: int, cut: int, end: int, layers_per_stage: int
) -> list[tuple[int, int, int]]:
    """``(first layer, end layer, rows)`` of each replay stage of layers
    ``cut``..``end``: a stage runs each request's last ``rows`` rows, as many as
    the layers from its first one on need for the last ``window`` rows to come
    out exact. The cut writes every row's KV, so the first stage keeps the rows
    of the layer after it."""
    stages = []
    first = cut
    while first < end:
        stop = min(end, max(first, cut + 1) + layers_per_stage)
        stages.append(
            (first, stop, exact_replay_rows(window, end - max(first, cut + 1)))
        )
        first = stop
    return stages


@dataclass
class ReplayStage:
    """A later replay stage: ``select`` picks its rows out of the previous stage's
    outputs, ``rows`` are their batch rows."""

    rows: torch.Tensor
    select: torch.Tensor
    forward_context: ForwardContext
    run_graph: Callable[..., tuple[torch.Tensor, ...]] | None = None


@dataclass
class ReplayBatch:
    """One forward's replay batch; ``run_graph`` is set when it runs in a graph.
    ``stages`` run after it on fewer rows each (layer-wise exact replay)."""

    rows: torch.Tensor
    forward_context: ForwardContext
    run_graph: Callable[..., tuple[torch.Tensor, ...]] | None = None
    stages: list[ReplayStage] = field(default_factory=list)


class DecoderReplayLayers:
    """Runs the replay layers on the step's replay batch.

    ``run_layers`` takes a batch's layer inputs and returns its per-row outputs;
    ``write_batch_kv`` takes them too and writes the first replay layer's KV for
    every row. ``metadata_prefixes`` are the attention metadata keys the layers
    read, so the replay batch builds only those. ``first_swa_prefix`` keys the
    first replay layer's SWA metadata, built with the batch's window starts.
    ``kept_rows`` is how many trailing rows of each prefill the layers run on:
    ``window`` by default, more when their outputs must match the full forward.

    With ``stage_rows`` the layers run in stages instead, each on fewer trailing
    rows (``exact_replay_stages``): ``run_stage(i, *inputs)`` runs stage ``i``
    and returns its output, pre-mix, post-mix, res-mix and residual (the last
    stage: its collapsed output and pre-mix), then the aux hidden states it
    captures. ``stage_metadata_prefixes`` are the metadata keys each stage reads.
    ``select_shared_rows(select)`` keeps the given rows of the per-row buffers
    layers share across stages (indexer top-k, candidate blocks).
    """

    def __init__(
        self,
        window: int,
        run_layers: Callable[..., tuple[torch.Tensor, ...]],
        write_batch_kv: Callable[..., None],
        metadata_prefixes: set[str],
        first_swa_prefix: str,
        kept_rows: int | None = None,
        stage_rows: list[int] | None = None,
        run_stage: Callable[..., tuple[torch.Tensor, ...]] | None = None,
        stage_metadata_prefixes: list[set[str]] | None = None,
        select_shared_rows: Callable[[torch.Tensor], None] | None = None,
    ) -> None:
        self.window = window
        self.kept_rows = window if kept_rows is None else kept_rows
        self.stage_rows = stage_rows or [self.kept_rows]
        assert self.stage_rows[0] == self.kept_rows
        assert len(self.stage_rows) == 1 or run_stage is not None
        self.run_stage = run_stage
        self.select_shared_rows = select_shared_rows
        self.stage_metadata_prefixes = stage_metadata_prefixes or [metadata_prefixes]
        self.run_layers = run_layers
        self.write_batch_kv = write_batch_kv
        self.metadata_prefixes = metadata_prefixes
        self.first_swa_prefix = first_swa_prefix
        # Set by the model state every step; None runs the layers on the batch.
        self.replay_batch: ReplayBatch | None = None
        # Set by the model state with replay graphs: PIECEWISE model graphs of
        # this many tokens or more break out to the replay batch.
        self.trim_threshold: int | None = None
        self._break_outputs: list[torch.Tensor] | None = None

    def __call__(
        self, hidden_states: torch.Tensor | MoEOutput, *states: torch.Tensor | None
    ) -> tuple[torch.Tensor, ...]:
        if self.uses_replay_batch():
            # Outside the break, so a PIECEWISE graph captures it.
            self.write_batch_kv(hidden_states, *states)
            if in_piecewise_cudagraph():
                return self._run_in_graph_break(hidden_states, *states)
        return self._run(hidden_states, *states)

    def uses_replay_batch(self) -> bool:
        """Whether this forward uses the replay batch: in a PIECEWISE graph by its
        size, fixed at capture (no replay batch is set then); else if one is set."""
        if in_piecewise_cudagraph():
            batch_descriptor = get_forward_context().batch_descriptor
            assert batch_descriptor is not None
            return self.graph_uses_replay_batch(batch_descriptor.num_tokens)
        return self.replay_batch is not None

    def graph_uses_replay_batch(self, num_tokens: int) -> bool:
        """Whether a PIECEWISE graph of ``num_tokens`` breaks out to the replay
        batch; capture and the model state's runtime replay batch share it."""
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
        if replay_batch.stages:
            return self._run_stages(replay_batch, hidden_states, *states)
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

    def _run_stages(
        self,
        replay_batch: ReplayBatch,
        hidden_states: torch.Tensor,
        *states: torch.Tensor | None,
    ) -> tuple[torch.Tensor, ...]:
        """Run the stages in turn, each on its rows of the previous one's outputs.
        Each stage's aux hidden states go to batch rows right away: a later
        stage's graph may reuse their memory."""
        positions, input_ids = states[0], states[1]
        num_tokens = hidden_states.shape[0]
        stages = [
            ReplayStage(
                replay_batch.rows,
                replay_batch.rows,
                replay_batch.forward_context,
                replay_batch.run_graph,
            ),
            *replay_batch.stages,
        ]
        aux: list[torch.Tensor] = []
        carried: tuple[torch.Tensor, ...] = ()
        for i, stage in enumerate(stages):
            if i == 0:
                sources = [(t, stage.rows) for t in (hidden_states, *states)]
            else:
                hidden, pre_mix, post_mix, res_mix, residual = carried
                sources = [
                    (hidden, stage.select),
                    (positions, stage.rows),
                    (input_ids, stage.rows),
                    (pre_mix, stage.select),
                    (post_mix, stage.select),
                    (res_mix, stage.select),
                    (residual, stage.select),
                ]
                if self.select_shared_rows is not None:
                    self.select_shared_rows(stage.select)
            with override_forward_context(stage.forward_context):
                if stage.run_graph is not None:
                    outputs = stage.run_graph(
                        [index for _, index in sources], [t for t, _ in sources]
                    )
                else:
                    assert self.run_stage is not None
                    outputs = self.run_stage(
                        i,
                        *(
                            None if t is None else t.index_select(0, index)
                            for t, index in sources
                        ),
                    )
            num_carried = 2 if i == len(stages) - 1 else 5
            aux.extend(
                _to_batch_rows(out, stage.rows, num_tokens)
                for out in outputs[num_carried:]
            )
            carried = tuple(outputs[:num_carried])
        last_rows = stages[-1].rows
        final = [_to_batch_rows(out, last_rows, num_tokens) for out in carried]
        return (*final, *aux)


def _to_batch_rows(
    out: torch.Tensor, rows: torch.Tensor, num_tokens: int
) -> torch.Tensor:
    # The rows outside the replay batch stay zero; nothing reads them.
    return out.new_zeros((num_tokens, *out.shape[1:])).index_copy_(0, rows, out)

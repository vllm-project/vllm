# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decoder-side SWA bounded replay: running the replay layers on their batch.

Layers past the last KV-source layer own nothing but their sliding-window KV,
so in prefill they run on each request's last ``window`` tokens only. The
model state builds that batch every step (``DeepseekV41ModelState``) and sets
``DecoderReplayLayers.replay_batch``; the layers gather their inputs by its
rows, run under its forward context, and scatter the outputs back to
full-batch rows.

Under piecewise (breakable) CUDA graphs the replay is an eager break of the
model graph, and the layers run in graphs of their own (``ReplayCudaGraphs``)
at the size the model state picked, on fixed input and output buffers.
"""

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import torch

from vllm.compilation.breakable_cudagraph import (
    BreakableCUDAGraphCapture,
    BreakableCUDAGraphWrapper,
    is_breakable_cudagraph_enabled,
)
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import (
    BatchDescriptor,
    DPMetadata,
    get_forward_context,
    override_forward_context,
)
from vllm.utils.torch_utils import weak_ref_tensor

# hidden_states, positions, input_ids, pre_mix, post_mix, res_mix, residual
States = tuple[torch.Tensor | None, ...]


@dataclass
class ReplayBatch:
    """The rows the replay layers run on, and what their forward context
    replaces in the batch's."""

    rows: torch.Tensor  # [num_tokens] rows of the full batch, in batch order
    trims: bool  # whether any request was cut down to its window
    graph_size: int | None  # padded size of the replay layers' graph; None: eager
    attn_metadata: dict[str, Any]
    slot_mapping: dict[str, torch.Tensor]
    is_padding: torch.Tensor
    dp_metadata: DPMetadata | None


class ReplayCudaGraphs:
    """Breakable graphs of the replay layers, keyed by padded replay size.

    Each is captured inside the model graph's capture of that size, while that
    capture is paused in the replay's eager break. The graph segments after the
    break read the replay's outputs, so inputs and outputs sit in fixed buffers,
    sized on the first (eager) forward.
    """

    def __init__(
        self,
        run_layers: Callable[..., tuple[torch.Tensor, ...]],
        vllm_config: VllmConfig,
        max_size: int,
    ) -> None:
        self.max_size = max_size
        self.wrapper = BreakableCUDAGraphWrapper(run_layers, vllm_config)
        self._inputs: list[torch.Tensor | None] | None = None
        self._outputs: list[torch.Tensor] | None = None

    @property
    def allocated(self) -> bool:
        return self._outputs is not None

    def allocate(self, states: States, outputs: tuple[torch.Tensor, ...]) -> None:
        size = self.max_size
        self._inputs = [
            None if t is None else t.new_zeros((size, *t.shape[1:])) for t in states
        ]
        self._outputs = [out.new_zeros((size, *out.shape[1:])) for out in outputs]

    def outputs(self, num_tokens: int) -> tuple[torch.Tensor, ...]:
        assert self._outputs is not None
        return tuple(out[:num_tokens] for out in self._outputs)

    def run(self, batch: ReplayBatch, states: States) -> tuple[torch.Tensor, ...]:
        """Run under the replay's forward context, whose batch descriptor is the
        graph's size."""
        assert self._inputs is not None
        num_tokens = batch.rows.shape[0]
        size = batch.graph_size
        assert size is not None and num_tokens <= size <= self.max_size
        desc = get_forward_context().batch_descriptor
        assert desc is not None and desc.num_tokens == size
        assert desc in self.wrapper.entries or BreakableCUDAGraphCapture.current(), (
            f"no replay graph of {size} rows; captured "
            f"{sorted(d.num_tokens for d in self.wrapper.entries)}"
        )
        for buf, t in zip(self._inputs, states):
            if buf is not None:
                torch.index_select(t, 0, batch.rows, out=buf[:num_tokens])
        inputs = [None if buf is None else buf[:size] for buf in self._inputs]
        outputs = self.wrapper(*inputs)
        return tuple(out[:num_tokens] for out in outputs)


class DecoderReplayLayers:
    """Runs the replay layers on the step's replay batch (see module doc).

    ``run_layers`` takes a batch's layer inputs (hidden states, positions, input
    ids, mHC states) and returns ``(hidden_states, pre_mix, *aux_hidden_states)``.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        window: int,
        source_attn: Any,
        run_layers: Callable[..., tuple[torch.Tensor, ...]],
    ) -> None:
        self.window = window
        self.run_layers = run_layers
        # Set by the model state for every forward that runs the replay layers
        # on fewer rows than the batch, or under a piecewise graph.
        self.replay_batch: ReplayBatch | None = None
        # Per-row indexer outputs the source publishes for the layers after it.
        self.row_buffers = [
            buf
            for buf in (
                source_attn.topk_indices_buffer,
                source_attn.candidate_block_buffer,
            )
            if buf is not None
        ]
        compilation_config = vllm_config.compilation_config
        self.graphs: ReplayCudaGraphs | None = None
        if (
            compilation_config.cudagraph_mode.has_piecewise_cudagraphs()
            and is_breakable_cudagraph_enabled()
            and compilation_config.cudagraph_capture_sizes
        ):
            self.graphs = ReplayCudaGraphs(
                run_layers, vllm_config, max(compilation_config.cudagraph_capture_sizes)
            )

    def __call__(self, *states: torch.Tensor | None) -> tuple[torch.Tensor, ...]:
        hidden_states = states[0]
        assert hidden_states is not None

        outer = BreakableCUDAGraphCapture.current()
        if outer is not None and outer.capturing:
            # The model graph's eager break: the segments after it read the
            # replay's outputs at fixed addresses.
            assert self.graphs is not None
            outputs = self.graphs.outputs(hidden_states.shape[0])
            weak_states = tuple(
                None if t is None else weak_ref_tensor(t) for t in states
            )
            outer.add_eager(lambda: self._run(weak_states, outputs))
            return outputs

        batch = self.replay_batch
        if batch is None or (
            batch.graph_size is not None
            and get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.NONE
        ):
            # No replay this forward, or the eager warmup of a graph capture.
            outputs = self.run_layers(*states)
        else:
            outputs = self._run(states)
        if self.graphs is not None and not self.graphs.allocated:
            self.graphs.allocate(states, outputs)
        return outputs

    def _run(
        self, states: States, outputs: tuple[torch.Tensor, ...] | None = None
    ) -> tuple[torch.Tensor, ...]:
        """Run the replay layers on the replay batch and scatter the results to
        full-batch rows, into ``outputs`` when given."""
        batch = self.replay_batch
        assert batch is not None, "piecewise forward without a replay batch"
        num_tokens = batch.rows.shape[0]

        forward_context = get_forward_context()
        replay_context = replace(
            forward_context,
            attn_metadata=batch.attn_metadata,
            slot_mapping=batch.slot_mapping,
            is_padding=batch.is_padding,
            dp_metadata=batch.dp_metadata or forward_context.dp_metadata,
            batch_descriptor=BatchDescriptor(num_tokens=batch.graph_size or num_tokens),
            cudagraph_runtime_mode=(
                CUDAGraphMode.NONE
                if batch.graph_size is None
                else CUDAGraphMode.PIECEWISE
            ),
        )
        with override_forward_context(replay_context):
            if batch.trims:
                for buf in self.row_buffers:
                    buf[:num_tokens].copy_(buf.index_select(0, batch.rows))
            if batch.graph_size is None:
                replay_outputs = self.run_layers(
                    *(
                        None if t is None else t.index_select(0, batch.rows)
                        for t in states
                    )
                )
            else:
                assert self.graphs is not None
                replay_outputs = self.graphs.run(batch, states)

        if outputs is None:
            if not batch.trims:
                return replay_outputs
            num_batch_tokens = states[0].shape[0]  # type: ignore[union-attr]
            outputs = tuple(
                t.new_zeros((num_batch_tokens, *t.shape[1:])) for t in replay_outputs
            )
        else:
            for out in outputs:
                out.zero_()
        for out, t in zip(outputs, replay_outputs):
            out.index_copy_(0, batch.rows, t)
        return outputs

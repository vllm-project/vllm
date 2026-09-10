# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decoder-side SWA bounded replay.

Layers past the last KV-source layer (the replay layers) own nothing but their
sliding-window KV, and decode reads only the trailing ``window`` positions of
it. In prefill they therefore run on each request's last ``window`` tokens (its
replay window), attending the source's full compressed KV plus their own window,
floored at the replay window's start since earlier positions hold no window KV
for these layers.

``ReplayBatchBuilder`` picks the rows (a ``ReplayInputBatch``),
``ReplayMetadataBuilder`` rebuilds the attention metadata for them,
``ReplayCudaGraphs`` runs the layers under graphs of their own when the model
itself runs under piecewise (breakable) graphs, and ``DecoderReplayLayers``
ties the three into the model's forward.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from vllm.compilation.breakable_cudagraph import (
    BreakableCUDAGraphCapture,
    BreakableCUDAGraphWrapper,
)
from vllm.config import VllmConfig
from vllm.forward_context import BatchDescriptor, get_forward_context
from vllm.logger import init_logger
from vllm.utils.torch_utils import weak_ref_tensor
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.buffer_utils import UvaBufferPool

logger = init_logger(__name__)

# hidden_states, positions, input_ids, pre_mix, post_mix, res_mix, residual
States = tuple[torch.Tensor | None, ...]


@dataclass
class ReplayInputBatch:
    """The batch the replay layers run on: each request's replay window rows."""

    rows: torch.Tensor  # [num_tokens] rows of the full batch
    query_start_loc: torch.Tensor  # [num_reqs + 1]
    query_start_loc_cpu: torch.Tensor
    replay_start: torch.Tensor  # [num_reqs] lowest window position per request
    positions: torch.Tensor | None  # [num_tokens]
    num_tokens: int
    max_query_len: int
    num_batch_tokens: int  # rows of the full batch, for scattering results back
    trims: bool  # False when every request already fits in the window

    def gather(self, t: torch.Tensor | None) -> torch.Tensor | None:
        return None if t is None else t.index_select(0, self.rows)

    def scatter(self, t: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        """Place replay rows back into a zero-filled full-batch tensor."""
        if out is None:
            out = t.new_zeros((self.num_batch_tokens, *t.shape[1:]))
        else:
            out.zero_()
        return out.index_copy_(0, self.rows, t)


class ReplayBatchBuilder:
    """Selects each request's replay window rows from the batch's metadata.

    The batch is laid out on the CPU each step and read by the GPU through UVA.
    """

    def __init__(self, window: int, max_num_tokens: int, max_num_reqs: int) -> None:
        self.window = window
        self._rows = UvaBufferPool(max_num_tokens, torch.int64)
        self._dropped = UvaBufferPool(max_num_reqs + 1, torch.int32)
        self._window_start = UvaBufferPool(max_num_reqs, torch.int32)

    @staticmethod
    def query_lens(common: Any) -> np.ndarray:
        return np.diff(common.query_start_loc_cpu[: common.num_reqs + 1].numpy())

    def trims(self, common: Any) -> bool:
        return bool((self.query_lens(common) > self.window).any())

    def build(self, common: Any, num_batch_tokens: int) -> ReplayInputBatch:
        num_reqs = common.num_reqs
        query_start_loc = common.query_start_loc_cpu[: num_reqs + 1].numpy()
        lens = self.query_lens(common)
        trimmed = lens > self.window
        keep = np.minimum(lens, self.window)
        new_query_start_loc = np.zeros(num_reqs + 1, dtype=np.int32)
        np.cumsum(keep, out=new_query_start_loc[1:])
        num_tokens = int(new_query_start_loc[-1])

        # Kept row r of request i is window_start[i] + (r - kept_start[i]).
        window_start = query_start_loc[1:] - keep
        kept_start = new_query_start_loc[:-1]
        rows = self._rows.copy_to_uva(
            np.repeat(window_start - kept_start, keep) + np.arange(num_tokens)
        )
        # Boundaries: the device ones minus the rows trimmed before them. Only
        # prefill rows are trimmed and their CPU lengths are exact; adaptive
        # verification resizes the leading decode requests on the GPU alone.
        dropped_before = self._dropped.copy_to_uva(
            query_start_loc - new_query_start_loc
        )
        replay_query_start_loc = common.query_start_loc[: num_reqs + 1] - dropped_before

        # A trimmed request holds no replay-layer window KV below its replay
        # window, on top of whatever the encoder-side replay already excludes.
        seq_lens = common.seq_lens_cpu_upper_bound[:num_reqs].numpy()
        replay_start = self._window_start.copy_to_uva(
            np.where(trimmed, seq_lens - self.window, 0)
        )
        if common.replay_start is not None:
            replay_start = torch.maximum(replay_start, common.replay_start[:num_reqs])

        return ReplayInputBatch(
            rows=rows,
            query_start_loc=replay_query_start_loc,
            query_start_loc_cpu=torch.from_numpy(new_query_start_loc),
            replay_start=replay_start,
            positions=None
            if common.positions is None
            else common.positions.index_select(0, rows),
            num_tokens=num_tokens,
            max_query_len=int(keep.max()),
            num_batch_tokens=num_batch_tokens,
            trims=bool(trimmed.any()),
        )


class ReplayMetadataBuilder:
    """Builds the replay layers' attention metadata for a batch.

    Private clones of the runner's builders do the building: the runner's back
    the full batch's metadata with persistent buffers, and that metadata
    outlives the forward (the runner hands it to the speculator).
    """

    def __init__(self, source_attn: Any, replay_attn: list[Any], max_num_tokens: int):
        self.compressed_prefix = source_attn.prefix
        self.indexer_prefix = source_attn.indexer.k_cache.prefix
        self.swa_prefixes = [attn.swa_cache_layer.prefix for attn in replay_attn]
        self._max_num_tokens = max_num_tokens
        self._builders: dict[Any, Any] = {}  # clones, by the runner's builder
        # Compacted slot mappings, one fixed buffer per KV-cache group (keyed
        # by the first layer seen in it): the window KV insert runs inside the
        # replay-layer graph and reads them by address.
        self._slot_mappings: dict[str, torch.Tensor] = {}

    def common(self, attn_metadata: Any) -> Any:
        """The batch's CommonAttentionMetadata, as the replay layers saw it."""
        assert isinstance(attn_metadata, dict)
        return attn_metadata[self.swa_prefixes[0]].common

    def build(
        self, attn_metadata: Any, batch: ReplayInputBatch, num_padded: int
    ) -> dict[str, Any]:
        """The metadata dict the replay layers run under, ``num_padded`` rows."""
        assert isinstance(attn_metadata, dict)
        replay = dict(attn_metadata)
        built: dict[Any, Any] = {}
        for prefix in self.swa_prefixes:
            full = attn_metadata[prefix]
            if full.builder not in built:
                built[full.builder] = self._builder_for(full).build(
                    0, self._compact(full, batch, num_padded, prefix)
                )
            replay[prefix] = built[full.builder]
        compressed = attn_metadata[self.compressed_prefix]
        source_common = self._compact(
            compressed, batch, num_padded, self.compressed_prefix
        )
        replay[self.compressed_prefix] = self._builder_for(compressed).build(
            0, source_common
        )
        # The indexer K cache shares the source's KV group, hence its metadata.
        indexer = attn_metadata[self.indexer_prefix]
        replay[self.indexer_prefix] = self._builder_for(indexer).build(0, source_common)
        return replay

    def _compact(
        self, full: Any, batch: ReplayInputBatch, num_padded: int, group: str
    ) -> Any:
        common = full.common
        slot_mapping = self._slot_mappings.get(group)
        if slot_mapping is None:
            slot_mapping = common.slot_mapping.new_empty(self._max_num_tokens)
            self._slot_mappings[group] = slot_mapping
        torch.index_select(
            common.slot_mapping, 0, batch.rows, out=slot_mapping[: batch.num_tokens]
        )
        if num_padded > batch.num_tokens:
            # Padding rows write nowhere, like the runner's own padding.
            slot_mapping[batch.num_tokens : num_padded] = PAD_SLOT_ID
        return common.replace_tokens(
            query_start_loc=batch.query_start_loc,
            query_start_loc_cpu=batch.query_start_loc_cpu,
            num_actual_tokens=batch.num_tokens,
            max_query_len=batch.max_query_len,
            slot_mapping=slot_mapping[:num_padded],
            replay_start=batch.replay_start,
            positions=batch.positions,
        )

    def _builder_for(self, full: Any) -> Any:
        src = full.builder
        if src not in self._builders:
            self._builders[src] = src.clone()
        return self._builders[src]


class ReplayCudaGraphs:
    """Breakable graphs of the replay layers, keyed by padded replay size.

    They are captured while the model graph is (the outer capture is paused in
    an eager break), on inputs and into outputs at fixed addresses.
    """

    def __init__(
        self,
        run_layers: Callable[..., tuple[torch.Tensor, ...]],
        vllm_config: VllmConfig,
        sizes: list[int],
    ) -> None:
        self.sizes = sorted(sizes)
        self.wrapper = BreakableCUDAGraphWrapper(run_layers, vllm_config)
        self._inputs: list[torch.Tensor | None] | None = None
        self._outputs: list[torch.Tensor] | None = None
        self._is_padding: torch.Tensor | None = None

    @property
    def allocated(self) -> bool:
        return self._outputs is not None

    def allocate(self, states: States, outputs: tuple[torch.Tensor, ...]) -> None:
        """Size the fixed input and output buffers from an eager forward."""
        size = self.sizes[-1]
        self._inputs = [
            None if t is None else t.new_zeros((size, *t.shape[1:])) for t in states
        ]
        self._outputs = [out.new_zeros((size, *out.shape[1:])) for out in outputs]
        self._is_padding = torch.zeros(size, dtype=torch.bool, device=outputs[0].device)

    def outputs(self, num_tokens: int) -> tuple[torch.Tensor, ...]:
        """Fixed-address outputs for the graph segments after the replay."""
        assert self._outputs is not None
        return tuple(out[:num_tokens] for out in self._outputs)

    def size_for(self, num_tokens: int) -> int | None:
        """Padded size whose graph exists or can be captured now, if any."""
        size = next((s for s in self.sizes if s >= num_tokens), None)
        if size is None:
            return None
        captured = BatchDescriptor(num_tokens=size) in self.wrapper.entries
        if captured or BreakableCUDAGraphCapture.current() is not None:
            return size
        return None

    def padding_mask(
        self, num_tokens: int, size: int, is_padding: torch.Tensor | None
    ) -> torch.Tensor:
        assert self._is_padding is not None
        mask = self._is_padding[:size]
        mask[:num_tokens] = False if is_padding is None else is_padding
        mask[num_tokens:] = True
        return mask

    def run(
        self, size: int, batch: ReplayInputBatch, states: States
    ) -> tuple[torch.Tensor, ...]:
        """Run the graph of ``size`` rows on the batch's rows of ``states``."""
        assert self._inputs is not None
        for buf, t in zip(self._inputs, states):
            if buf is not None:
                torch.index_select(t, 0, batch.rows, out=buf[: batch.num_tokens])
        inputs = [None if buf is None else buf[:size] for buf in self._inputs]
        outputs = self.wrapper.run(BatchDescriptor(num_tokens=size), *inputs)
        return tuple(out[: batch.num_tokens] for out in outputs)


class DecoderReplayLayers:
    """Runs the replay layers on each request's replay window (see module doc).

    ``run_layers`` takes a batch's layer inputs (hidden states, positions, input
    ids, mHC states) and returns ``(hidden_states, pre_mix, *aux_hidden_states)``.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        window: int,
        source_attn: Any,
        replay_attn: list[Any],
        run_layers: Callable[..., tuple[torch.Tensor, ...]],
        graph_sizes: list[int],
    ) -> None:
        logger.info_once(
            "Decoder SWA bounded replay: layers past the last KV source prefill "
            "only each request's last %d tokens.",
            window,
        )
        self.run_layers = run_layers
        # Per-row indexer outputs the source publishes for the layers after it.
        self.row_buffers = [
            buf
            for buf in (
                source_attn.topk_indices_buffer,
                source_attn.candidate_block_buffer,
            )
            if buf is not None
        ]
        scheduler_config = vllm_config.scheduler_config
        max_num_tokens = scheduler_config.max_num_batched_tokens
        self.batch_builder = ReplayBatchBuilder(
            window, max_num_tokens, scheduler_config.max_num_seqs
        )
        self.metadata = ReplayMetadataBuilder(source_attn, replay_attn, max_num_tokens)
        self.graphs = (
            ReplayCudaGraphs(run_layers, vllm_config, graph_sizes)
            if graph_sizes
            else None
        )

    def __call__(self, *states: torch.Tensor | None) -> tuple[torch.Tensor, ...]:
        hidden_states = states[0]
        assert hidden_states is not None
        num_tokens = hidden_states.shape[0]

        outer = BreakableCUDAGraphCapture.current()
        if outer is not None and outer.capturing:
            # Piecewise capture of the whole model: the replay is an eager break,
            # so each replay plans from its own batch. Its outputs must sit at
            # fixed addresses for the graph segments after it.
            assert self.graphs is not None
            outputs = self.graphs.outputs(num_tokens)
            weak_states = tuple(weak_ref_tensor(t) for t in states)
            outer.add_eager(lambda: self._run(weak_states, outputs))
            return outputs

        attn_metadata = get_forward_context().attn_metadata
        common = (
            self.metadata.common(attn_metadata)
            if isinstance(attn_metadata, dict)
            else None
        )
        if common is not None and self.batch_builder.trims(common):
            assert not torch.cuda.is_current_stream_capturing()
            outputs = self._run(states)
        else:
            # Decode batches (including full-graph captures) and metadata-less
            # warmup runs.
            outputs = self.run_layers(*states)
        if self.graphs is not None and not self.graphs.allocated:
            # The first forward is the runner's eager profile run.
            assert not torch.cuda.is_current_stream_capturing()
            self.graphs.allocate(states, outputs)
        return outputs

    def _run(
        self, states: States, outputs: tuple[torch.Tensor, ...] | None = None
    ) -> tuple[torch.Tensor, ...]:
        """Run the replay layers on this batch's replay rows and scatter the results
        back to full-batch rows, into ``outputs`` when given."""
        hidden_states = states[0]
        assert hidden_states is not None
        forward_context = get_forward_context()
        batch = self.batch_builder.build(
            self.metadata.common(forward_context.attn_metadata), hidden_states.shape[0]
        )
        replay_outputs = self._run_replay(batch, states)
        if outputs is None:
            return tuple(batch.scatter(t) for t in replay_outputs)
        for out, t in zip(outputs, replay_outputs):
            batch.scatter(t, out=out)
        return outputs

    def _run_replay(
        self, batch: ReplayInputBatch, states: States
    ) -> tuple[torch.Tensor, ...]:
        forward_context = get_forward_context()
        saved = (forward_context.attn_metadata, forward_context.is_padding)
        size = self.graphs.size_for(batch.num_tokens) if self.graphs else None
        try:
            # A graph always runs on rebuilt metadata, so what its captured
            # kernels read by address is this step's data.
            if batch.trims or size is not None:
                forward_context.attn_metadata = self.metadata.build(
                    saved[0], batch, size or batch.num_tokens
                )
            if batch.trims:
                for buf in self.row_buffers:
                    buf[: batch.num_tokens].copy_(batch.gather(buf))
            is_padding = batch.gather(saved[1])
            if size is None:
                forward_context.is_padding = is_padding
                return self.run_layers(*(batch.gather(t) for t in states))
            assert self.graphs is not None
            forward_context.is_padding = self.graphs.padding_mask(
                batch.num_tokens, size, is_padding
            )
            return self.graphs.run(size, batch, states)
        finally:
            forward_context.attn_metadata, forward_context.is_padding = saved

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The specialized decode executor.

`SpecializedDecode` runs the decoder layers and the final hyper-connection
mixer of `Qwen4ExpModel` for the steps captured into FULL CUDA graphs of pure
decode batches of up to `DecodeConfig.max_rows` rows. Its model state builds
it once the weights are loaded. `plan` decides per forward whether it runs;
the model's forward then passes the returned `DecodeStep` to the backbone,
which embeds the tokens and hands the layers to `DecodeStep.run`. Every
other forward (prefill, mixed, eager, PIECEWISE) runs the stock layers, which
the executor leaves unchanged. At TP > 1 the executor also replicates the
token embedding's table (`embedding.py`): every lookup captured into a CUDA
graph, the target's and the MTP drafter's, reads it without an all-reduce.

`run` is the stock layer loop and final mixer of `Qwen4ExpModel.forward`
(over `Qwen4ExpDecoderLayer.forward` without sequence parallelism) with the
same submodule calls, except for:

- the hyper-connection reads (`GatedResidual.mix` and `combine_and_mix`)
  of steps of up to 64 rows, and at TP1 of up to 128 rows: those run the
  read kernels of `..hc` (`hc_read`; at TP1 the tensor-core kernels from
  `TC_MIN_ROWS` rows, unless the device cannot run them,
  `use_tensor_core_reads`). Larger steps run the stock reads, except the
  reads of the row split below.
- the GDN layers' core (`GdnLayers`, `gdn.py`): the layer's own
  projections around the fused GDN decode core of `..gdn` for speculative
  decode steps it covers, the stock layer otherwise.
- the sublayer boundaries (`Boundaries`, `boundary.py`): at TP > 1 the
  attention and MoE blocks, at TP 1 the MoE blocks, leave their all-reduce
  (and for MoE the shared expert's gate and the routed add) to one kernel at
  the start of the next read, which also runs the read's combine-norm.
- the row split (`RowSplit`, `split.py`): in TP4 steps of 32 to 256 rows,
  from the MLP read of the last PLE layer on, each rank keeps its quarter of
  the rows of the streams (at most 64, which the read kernels take); the
  boundary kernel reduce-scatters the blocks' outputs and the reads all-gather
  the block inputs, and the final mixer's streams are gathered into the MTP
  drafter's input buffer.
- the layers' BF16 projections (`PlannedGemms`, `gemm.py`): the skinny
  GEMM plans of the specialized decode's own table at the row counts it
  lists, the layer's own GEMM otherwise.
- the PLE layer's lookup at TP > 1 (`PleGathers`, `ple.py`): the Lamport
  row gather of `..ple` in place of the int8 all-reduce, copy and dequant.
- the QSA layers' sparse attention, for BF16 main K/V caches (`QsaRows`,
  `qsa.py`): without the stock zero fill of the step's rows, and in steps
  whose launch has `MIN_CTAS` (256) CTAs or more, 128 rows at TP1, the row
  attention of `..qsa` instead of the stock split-K kernel and merge.
- with the overlapped metadata replay bound (`metadata`), the capture's
  waits for it (`CaptureWaits`, `decode_metadata.py`): before the first
  GDN layer and before the first PLE or QSA layer.

Whether a step is a pure decode is read from the attention metadata of every
KV-cache group the layers use, all host fields that every TP rank shares:

- GDN linear attention (`GDNAttentionMetadata`), the PLE short convolution
  (`PleShortConvAttentionMetadata`) and the QSA indexer's raw and compressed
  key caches (`QSAIndexerMetadata`) count the step's prefill requests in
  `num_prefills`. A speculative verification is a decode for all three; GDN
  and the short convolution also count a request's first chunk as a prefill
  even when it is one token long.
- The QSA main cache (`Qwen4ExpQSAMetadata`) carries only the step's longest
  query, `max_query_len`, which in a decode step is at most one request's
  verification (the MTP draft tokens plus one); above it the QSA layer takes
  its prefill configuration.
"""

import time
from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, NamedTuple, cast

import torch

from vllm.config import CUDAGraphMode
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.attention.backends.short_conv_attn import PleShortConvAttentionMetadata

from ....common.qsa import Qwen4ExpQSAMetadata
from ....common.qsa_indexer import QSAIndexerMetadata
from .. import DecodeConfig
from .boundary import Boundaries, PendingBlock, Project
from .decode_metadata import CaptureWaits, DecodeMetadataGraphs
from .embedding import ReplicatedEmbedding, replicate_embedding
from .gdn import GdnLayers
from .gemm import PlannedGemms
from .ple import PleGathers
from .qsa import QsaRows
from .split import RowSplit, split_max_rows, wants_split

if TYPE_CHECKING:
    from ...hyperconnection import GatedResidual
    from ...model import Qwen4ExpDecoderLayer, Qwen4ExpModel

logger = init_logger(__name__)

# A hyper-connection read: the module, the streams, the pending block output
# and injection logits; returns the streams, the block input and the new
# injection logits (``GatedResidual.combine_and_mix``).
HcRead = Callable[
    ["GatedResidual", torch.Tensor, torch.Tensor | None, torch.Tensor | None],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor | None],
]

# The metadata types that count a step's prefill requests in num_prefills.
PREFILL_COUNTING_METADATA = (
    GDNAttentionMetadata,
    PleShortConvAttentionMetadata,
    QSAIndexerMetadata,
)


class PureDecodeGate(NamedTuple):
    """The attention metadata that decides whether a step is a pure decode."""

    # Layers whose metadata counts the step's prefills (num_prefills): GDN,
    # PLE short convolution and QSA indexer caches.
    prefill_counting_layers: tuple[str, ...]
    # QSA main caches, whose metadata carries the step's longest query.
    qsa_main_layers: tuple[str, ...]
    # The longest query of a decode step: one request's verification.
    max_decode_query_len: int

    def is_pure_decode(self, attn_metadata: dict[str, AttentionMetadata]) -> bool:
        """Whether every layer's metadata in `attn_metadata` (the forward
        context's, by layer name) describes a step without prefills; False
        also when a layer's metadata is missing or of another type."""
        for name in self.prefill_counting_layers:
            metadata = attn_metadata.get(name)
            if (
                not isinstance(metadata, PREFILL_COUNTING_METADATA)
                or metadata.num_prefills > 0
            ):
                return False
        for name in self.qsa_main_layers:
            metadata = attn_metadata.get(name)
            if (
                not isinstance(metadata, Qwen4ExpQSAMetadata)
                or metadata.max_query_len > self.max_decode_query_len
            ):
                return False
        return True

    def captures_pure_decode(self) -> bool:
        """Whether the forward being captured into a CUDA graph is a FULL graph
        of a pure decode step: not a PIECEWISE capture (Model Runner V2
        captures FULL graphs with the forward context's mode NONE, and
        PIECEWISE ones, which report capturing too, with PIECEWISE), and
        metadata without prefills for every KV-cache group of the layers."""
        context = get_forward_context()
        attn_metadata = context.attn_metadata
        if context.cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE or not isinstance(
            attn_metadata, dict
        ):
            return False
        return self.is_pure_decode(attn_metadata)


def stock_read(
    hc: "GatedResidual",
    hidden_states: torch.Tensor,
    block_output: torch.Tensor | None,
    injection: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """The stock hyper-connection read: `hc.mix`, or `hc.combine_and_mix`
    of a pending `block_output`."""
    if block_output is None:
        return hc.mix(hidden_states)
    return hc.combine_and_mix(hidden_states, block_output, injection)


def stock_project(
    hc: "GatedResidual", xn: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """The stock projections of a read from its normalized streams `xn`
    (what `hc.combine_and_mix` runs after its combine-norm): the block input
    and the injection logits."""
    lowrank, injection = hc._down_and_inject(xn)
    return hc._up_gate_mix(lowrank, xn), injection


class DecodeStep(NamedTuple):
    """The host-side gates of one captured decode step, by its rows."""

    rows: int
    executor: "SpecializedDecode"
    # The step's attention metadata by layer name (the forward context's);
    # None runs every GDN layer stock.
    attn_metadata: dict[str, AttentionMetadata] | None = None

    def run(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
        query_start_loc: torch.Tensor | None,
        ngram_context: torch.Tensor | None,
        deepstack_input_embeds: IntermediateTensors | None,
    ) -> torch.Tensor:
        """The backbone's output after its embedding, from the embedded
        multi-stream `hidden_states` and the inputs of the stock layer loop
        (`SpecializedDecode.run`)."""
        return self.executor.run(
            self,
            hidden_states,
            positions,
            input_ids,
            query_start_loc,
            ngram_context,
            deepstack_input_embeds,
        )


class SpecializedDecode:
    """The decode executor over one backbone's local layers."""

    def __init__(self, backbone: "Qwen4ExpModel", config: DecodeConfig) -> None:
        """The executor of the target model's `backbone`, once its weights
        are loaded; raises on layers whose decode steps the per-step gate
        cannot recognize."""
        # Imported here: the QSA layer module imports the model module, and
        # the read kernels are CuTe DSL kernels (CUDA builds only).
        from ...qsa import Qwen4ExpQSAAttention
        from ..hc.read import (
            MAX_ROWS,
            TC_MAX_ROWS,
            hc_project,
            hc_read,
            use_tensor_core_reads,
        )

        self.backbone = backbone
        self.config = config
        self.layers = [
            cast("Qwen4ExpDecoderLayer", backbone.layers[index])
            for index in range(backbone.start_layer, backbone.end_layer)
        ]
        # The read kernels and the rows of the largest step they serve: at
        # TP1 with the tensor-core kernels from TC_MIN_ROWS rows.
        self.tensor_core_reads: bool = use_tensor_core_reads(
            config.max_rows,
            get_tensor_model_parallel_world_size(),
            torch.accelerator.current_device_index(),
        )
        self.kernel_read: HcRead = partial(hc_read, tensor_cores=self.tensor_core_reads)
        self.kernel_project: Project = partial(
            hc_project, tensor_cores=self.tensor_core_reads
        )
        self.kernel_read_rows = min(
            config.max_rows, TC_MAX_ROWS if self.tensor_core_reads else MAX_ROWS
        )
        prefill_counting: list[str] = []
        qsa_main: list[str] = []
        for layer in self.layers:
            if layer.ple is not None:
                prefill_counting.append(layer.ple.prefix)
            if layer.layer_type == "linear_attention":
                prefill_counting.append(layer.linear_attn.prefix)
            elif isinstance(layer.self_attn, Qwen4ExpQSAAttention):
                indexer = layer.self_attn.indexer
                prefill_counting.append(indexer.raw_key_cache.prefix)
                prefill_counting.append(indexer.compressed_key_cache.prefix)
                qsa_main.append(layer.self_attn.layer_name)
            else:
                raise ValueError(
                    "decode_backend=latency_specialized needs QSA and GDN "
                    f"attention layers, not {type(layer.self_attn).__name__}"
                )
        self.gate = PureDecodeGate(
            prefill_counting_layers=tuple(prefill_counting),
            qsa_main_layers=tuple(qsa_main),
            max_decode_query_len=config.tokens_per_request,
        )
        self.gdn = GdnLayers.build(
            [
                layer.linear_attn
                for layer in self.layers
                if layer.layer_type == "linear_attention"
            ],
            config.tokens_per_request,
            config.max_rows // config.tokens_per_request,
        )
        # The rows of the largest step whose reads can split among the TP
        # ranks: each rank then reads at most the read kernels' 64 rows.
        split_rows = split_max_rows(config.max_rows)
        # Collective over the TP group: every rank maps the others' mailboxes.
        self.boundaries = Boundaries.build(
            self.layers, config.max_rows, split=wants_split(split_rows)
        )
        # Collective over the TP group; None when every step reads replicated.
        self.split: RowSplit | None = RowSplit.build(
            self.layers, self.boundaries, split_rows
        )
        self.gemms = PlannedGemms.build(self.layers)
        self.qsa = QsaRows.build(self.layers)
        # Collective over the PLE tables' groups (the TP group).
        self.ple = PleGathers.build(self.layers, config.max_rows)
        # Collective over the TP group; None at TP 1. The target's embedding
        # module, which the MTP drafter shares, takes the table from now on.
        self.embedding: ReplicatedEmbedding | None = replicate_embedding(
            backbone.embed_tokens
        )
        # The replayed attention metadata whose side-stream replays the
        # captured steps wait for; the model state binds it.
        self.metadata: DecodeMetadataGraphs | None = None

    def warm_up(self) -> None:
        """Run every variant of the read kernels once (`warm_up_hc_read`),
        of the fused GDN decode core (`GdnLayers.warm_up`), of the boundary
        kernel (`Boundaries.warm_up`, collective over the TP group), the
        row split's gathers (`RowSplit.warm_up`, collective), every planned
        GEMM (`PlannedGemms.warm_up`), the PLE row gather
        (`PleGathers.warm_up`, collective) and the QSA row attention
        (`QsaRows.warm_up`), so that no CUDA graph capture compiles one. The
        model state calls it right after building the executor: memory
        profiling already captures the largest decode graphs, which take the
        executor."""
        # Imported here, as in __init__: CuTe DSL kernels.
        from ..hc.read import TC_MIN_ROWS, warm_up_hc_read

        final_mixer = self.backbone.hyper_connection_mixer
        assert final_mixer is not None
        tensor_cores = (
            f" (tensor cores from {TC_MIN_ROWS} rows)" if self.tensor_core_reads else ""
        )
        start = time.perf_counter()
        with torch.inference_mode():
            warm_up_hc_read(
                self.layers[0].mlp_hyper_connection,
                final_mixer,
                self.kernel_read_rows,
                self.tensor_core_reads,
            )
        logger.info_once(
            "Warmed up the Qwen4Exp specialized decode's hyper-connection read "
            "kernels for 1-%d rows%s in %.1f s.",
            self.kernel_read_rows,
            tensor_cores,
            time.perf_counter() - start,
        )
        start = time.perf_counter()
        with torch.inference_mode():
            schedules = self.gdn.warm_up()
        if schedules:
            logger.info_once(
                "Warmed up the Qwen4Exp specialized decode's GDN decode core for "
                "1-%d requests (%s) in %.1f s.",
                self.gdn.max_requests,
                ", ".join(str(schedule) for schedule in schedules),
                time.perf_counter() - start,
            )
        start = time.perf_counter()
        with torch.inference_mode():
            self.boundaries.warm_up(self.layers[0].mlp_hyper_connection)
        logger.info_once(
            "Warmed up the Qwen4Exp specialized decode's boundary kernel in %.1f s.",
            time.perf_counter() - start,
        )
        if self.split is not None:
            start = time.perf_counter()
            with torch.inference_mode():
                self.split.warm_up()
            logger.info_once(
                "Warmed up the Qwen4Exp specialized decode's row-split gathers in "
                "%.1f s.",
                time.perf_counter() - start,
            )
        start = time.perf_counter()
        planned = self.gemms.warm_up()
        if planned:
            logger.info_once(
                "Warmed up the Qwen4Exp specialized decode's %d GEMM plans for "
                "%d layers in %.1f s.",
                len(planned),
                len(self.gemms.sites),
                time.perf_counter() - start,
            )
        start = time.perf_counter()
        with torch.inference_mode():
            gathers = self.ple.warm_up()
        if gathers:
            logger.info_once(
                "Warmed up the Qwen4Exp specialized decode's PLE row gather for %d "
                "layer(s) in %.1f s.",
                gathers,
                time.perf_counter() - start,
            )
        start = time.perf_counter()
        shapes = self.qsa.warm_up()
        if shapes:
            logger.info_once(
                "Warmed up the Qwen4Exp specialized decode's QSA row attention "
                "and select for %d layers in %.1f s.",
                len(self.qsa.layers),
                time.perf_counter() - start,
            )

    def plan(
        self,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
    ) -> DecodeStep | None:
        """The step this forward runs, or None to run the stock layers: only
        FULL CUDA graph captures of pure decode steps of up to
        `DecodeConfig.max_rows` rows run the executor. The last axis of
        `positions` (MRoPE positions have a leading axis of 3) is the step's
        rows; a pipeline stage's `intermediate_tensors` always run stock."""
        rows = positions.shape[-1]
        if (
            intermediate_tensors is not None
            or not 0 < rows <= self.config.max_rows
            or not is_forward_context_available()
            # Eager steps, profiling and the eager run before each capture run
            # stock (a replayed graph does not call forward).
            or not torch.cuda.is_current_stream_capturing()
            or not self.gate.captures_pure_decode()
        ):
            self._note_stock_capture(rows)
            return None
        return DecodeStep(
            rows=rows,
            executor=self,
            attn_metadata=cast(
                "dict[str, AttentionMetadata]", get_forward_context().attn_metadata
            ),
        )

    def _note_stock_capture(self, rows: int) -> None:
        """Tell the GDN records protocol (`gdn_records.py`) and the replayed
        metadata (`DecodeMetadataGraphs.captured_stock`) that a FULL CUDA
        graph capture of `rows` rows runs the stock layers (a forward that
        `plan` refuses while capturing, not a PIECEWISE one)."""
        if not (
            torch.cuda.is_current_stream_capturing()
            and is_forward_context_available()
            and get_forward_context().cudagraph_runtime_mode != CUDAGraphMode.PIECEWISE
        ):
            return
        records = self.gdn.records
        if records is not None:
            records.finish_capture(rows)
        if self.metadata is not None:
            self.metadata.captured_stock(rows)

    def run(
        self,
        step: DecodeStep,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
        query_start_loc: torch.Tensor | None,
        ngram_context: torch.Tensor | None,
        deepstack_input_embeds: IntermediateTensors | None,
    ) -> torch.Tensor:
        """The decoder layers and the final mixer of `step`, on the embedded
        multi-stream `hidden_states`; returns what `Qwen4ExpModel.forward`
        returns on the last pipeline rank. The stock loop, PLE prefetches and
        submodule calls in the stock order, with the read kernels for the
        hyper-connection reads of steps of up to `kernel_read_rows` rows,
        `GdnLayers` for the GDN layers, `Boundaries` for the blocks'
        all-reduces and the next reads' combine-norms, the layers' GEMM
        plans (`PlannedGemms`) and the QSA attention without the fill of the
        step's rows, with the row attention in steps of `MIN_CTAS` CTAs and
        more (`QsaRows`), the PLE row gather at TP > 1 (`PleGathers`) and
        the row split in large TP4 steps (`RowSplit`). With the GDN records
        protocol, the capture tells it which GDN layers launched the records
        core."""
        records = self.gdn.records
        if records is not None and not torch.cuda.is_current_stream_capturing():
            records = None
        if records is not None:
            records.start_capture()
        with self.gemms.active(), self.qsa.active(step.rows), self.ple.active():
            output = self._run_layers(
                step,
                hidden_states,
                positions,
                input_ids,
                query_start_loc,
                ngram_context,
                deepstack_input_embeds,
            )
        if records is not None:
            records.finish_capture(step.rows)
        return output

    def _run_layers(
        self,
        step: DecodeStep,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
        query_start_loc: torch.Tensor | None,
        ngram_context: torch.Tensor | None,
        deepstack_input_embeds: IntermediateTensors | None,
    ) -> torch.Tensor:
        """`run`'s layer loop and final mixer."""
        backbone = self.backbone
        layers = self.layers
        kernel_reads = step.rows <= self.kernel_read_rows
        read = self.kernel_read if kernel_reads else stock_read
        project = self.kernel_project if kernel_reads else stock_project
        boundaries = self.boundaries
        # Deepstack embeddings add to the MoE outputs before the next read.
        moe_boundaries = deepstack_input_embeds is None
        attn_metadata = step.attn_metadata or {}
        block_output: torch.Tensor | PendingBlock | None = None
        injection: torch.Tensor | None = None
        # The waits for the overlapped metadata replay: for the GDN groups'
        # metadata before the first GDN layer, for all of it before the first
        # PLE or QSA layer, and at the end at the latest.
        waits = CaptureWaits(self.metadata)
        # The row split (``split.py``): from the MLP read of the layer at
        # ``split.start`` on, the streams and injection logits are this rank's
        # rows (``sharded``).
        split = self.split
        if split is not None and not split.takes(step.rows, not moe_boundaries):
            split = None
        sharded = False

        def read_block(
            hc: "GatedResidual",
            hidden_states: torch.Tensor,
            block_output: torch.Tensor | PendingBlock | None,
            injection: torch.Tensor | None,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
            """The read of `hc` after `block_output`: the boundary kernel
            for a pending block (`sharded`: its split call), `read`
            otherwise."""
            if isinstance(block_output, PendingBlock):
                if sharded:
                    assert split is not None
                    return boundaries.read(
                        hc,
                        hidden_states,
                        block_output,
                        injection,
                        split.project,
                        split=True,
                    )
                return boundaries.read(
                    hc, hidden_states, block_output, injection, project
                )
            # Every read of a split step after its first is a boundary's
            # (``split_unsupported``).
            assert not sharded
            return read(hc, hidden_states, block_output, injection)

        backbone._start_layer_ple_prefetch(
            layers[0], input_ids, query_start_loc, ngram_context
        )
        for position, layer in enumerate(layers):
            if position + 1 < len(layers):
                backbone._start_layer_ple_prefetch(
                    layers[position + 1], input_ids, query_start_loc, ngram_context
                )
            attn_hc = layer.attn_hyper_connection
            if layer.ple is not None:
                # PLE adds to the multi-stream state: materialize the pending
                # combine first (the MoE before a PLE layer reduces its own
                # output).
                assert not isinstance(block_output, PendingBlock)
                if block_output is not None:
                    hidden_states = attn_hc.combine(
                        hidden_states, block_output, injection
                    )
                    block_output = injection = None
                waits.before_all()
                hidden_states = layer.ple(
                    hidden_states, input_ids, query_start_loc, ngram_context
                )
            hidden_states, block_input, injection = read_block(
                attn_hc, hidden_states, block_output, injection
            )
            if layer.layer_type == "linear_attention":
                gdn = layer.linear_attn
                waits.before_gdn()
                with boundaries.attention_partial(gdn.out_proj) as partial:
                    output = self.gdn(gdn, block_input, attn_metadata.get(gdn.prefix))
            else:
                waits.before_all()
                with boundaries.attention_partial(layer.self_attn.o_proj) as partial:
                    output = layer.self_attn(
                        hidden_states=block_input, positions=positions
                    )
            attn_output = PendingBlock(output) if partial else output
            if split is not None and position == split.start:
                # The first split read: keep this rank's rows from here on.
                assert injection is not None
                hidden_states = split.own(hidden_states)
                injection = split.own(injection)
                sharded = True
            hidden_states, block_input, injection = read_block(
                layer.mlp_hyper_connection, hidden_states, attn_output, injection
            )
            pending = boundaries.moe_output(position, block_input, moe_boundaries)
            block_output = layer.mlp(block_input) if pending is None else pending
            if deepstack_input_embeds is not None:
                assert isinstance(block_output, torch.Tensor)
                hidden_states, block_output, injection = self._add_deepstack_embeds(
                    position,
                    hidden_states,
                    block_output,
                    injection,
                    deepstack_input_embeds,
                )
        final_mixer = backbone.hyper_connection_mixer
        assert final_mixer is not None
        multi_hidden, sample_hidden_states, _ = read_block(
            final_mixer, hidden_states, block_output, injection
        )
        # The MTP drafter's multi-stream input, as the stock forward fills it
        # (a split step: every rank's rows, gathered into it).
        if backbone._mtp_hidden_buffer is not None:
            if sharded:
                assert split is not None
                split.streams(
                    multi_hidden,
                    backbone._mtp_hidden_buffer[: sample_hidden_states.shape[0]],
                )
            else:
                backbone._mtp_hidden_buffer[: multi_hidden.shape[0]].copy_(multi_hidden)
        waits.finish(step.rows)
        return sample_hidden_states

    def _add_deepstack_embeds(
        self,
        position: int,
        hidden_states: torch.Tensor,
        block_output: torch.Tensor,
        injection: torch.Tensor | None,
        deepstack_input_embeds: IntermediateTensors,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Add the deepstack embedding of the layer at `position`, if it has
        one, to the multi-stream state after it, as the stock forward does:
        the layer's pending combine (`block_output`, `injection`) is
        materialized first, since deepstack ends the delayed combine. Returns
        the state and the pending combine as the loop carries them. Without
        sequence parallelism, which the deployment check rejects."""
        backbone = self.backbone
        layer_idx = backbone.start_layer + position
        if layer_idx >= len(deepstack_input_embeds):
            return hidden_states, block_output, injection
        deepstack_embed = deepstack_input_embeds[f"deepstack_input_embeds_{layer_idx}"]
        deepstack_embed = (
            deepstack_embed.unsqueeze(-2)
            .expand(
                *deepstack_embed.shape[:-1],
                backbone.config.hc_count,
                backbone.config.hidden_size,
            )
            .flatten(-2)
        )
        hidden_states = self.layers[position].mlp_hyper_connection.combine(
            hidden_states, block_output, injection
        )
        return hidden_states + deepstack_embed, None, None

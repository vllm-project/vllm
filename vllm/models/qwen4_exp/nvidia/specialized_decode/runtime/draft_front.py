# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The front of the MTP draft passes in the specialized decode.

Each draft pass of Qwen4Exp's MTP model (`Qwen4ExpMultiTokenPredictor`)
starts by mixing the new token's embedding `e` with the four
hyper-connection streams `h` of the previous pass::

    e' = fc_embedding(pre_fc_norm_embedding(e))          [rows, 2560]
    h' = fc_hidden(pre_fc_norm_hidden(h)), per stream    [rows, 4, 2560]

The stock front runs each `GemmaRMSNorm` as about ten PyTorch kernels, and
each column-parallel 2560 x 2560 projection as a GEMM of the rank's output
columns plus an all-gather and a copy at TP4: about 90-100 us of each of the
three draft passes of a TP4 step and 55-65 us at TP1. `DraftFront` runs the
two norms as one CuTe DSL kernel (`..mtp.front_norm`) and both projections
on full copies of their weights, so no rank gathers anything.

Replicated weights. At TP > 1 the front keeps a full `[2560, 2560]` copy of
both projection weights besides the stock shards (26 MB per rank): each
weight's loader is wrapped so that the checkpoint tensor, which arrives whole,
is copied into the replica before the stock loader takes the rank's shard.
The replicas are allocated with the model, so memory profiling counts them. At
TP1 the front uses the layers' own weights.

Binding. The front binds its projection weights at the end of the draft
model's `load_weights` (`finish_loading`) and refuses until then. Loaders
that never call `load_weights` (dummy weights, the weight cache, sharded
state) leave it unbound: every pass runs stock, and the first capture that
would have taken the front logs why. If either replica was not loaded from
the checkpoint (a loader that never sees the whole tensor), `finish_loading`
refuses, logs why and frees the replicas.

Which passes. The front runs inside FULL CUDA graph captures of pure decode
steps only: the draft's first pass of a decode step (`num_speculative_tokens
+ 1` rows a request) and its later passes (one row a request), for up to the
executor's largest step (`DecodeConfig.max_rows`). Whether a step is a pure
decode is read from the draft layer's QSA metadata, host fields that every TP
rank shares (`PureDecodeGate`). Every other forward (prefill, mixed, eager,
PIECEWISE, the eager run before each capture) runs the stock front, as does
every forward of a deployment the front refuses at load (logged once). Since
the front drops the stock all-gathers, all TP ranks must take the same path in
every pass; they do, as every input to the decision is shared host state.

The draft layer. A pass that takes the front also runs the draft layer and
the final mixer as the executor runs a target layer (`decode.py`):

- its BF16 projections (QSA's, the MoE router's and the shared expert's) with
  the specialized decode's GEMM plans (`PlannedGemms`, `gemm.py`) and its
  QSA attention without the fill of the pass's rows, with the row attention in
  passes whose launch has `MIN_CTAS` (256) CTAs and more (the first pass of a
  TP1 step of 32 requests and more) (`QsaRows`, `qsa.py`);
- its three hyper-connection reads (the attention block's, which combines the
  projected embedding into every stream with unit weight, the MoE block's and
  the final mixer's) with the read kernels of `..hc` (`hc_read`) in passes
  of up to 64 rows, the stock reads in larger ones;
- its sublayer boundaries with the boundary kernel (`Boundaries`,
  `boundary.py`): at TP > 1 the attention block, and the MoE block when its
  layer passes `moe_unsupported`, leave their all-reduce (and for MoE the
  shared expert's gate and the routed add) to the kernel at the start of the
  next read. The draft layer has its own `Boundaries`, with its own kernel
  instance, mailbox and generation words, built at the end of loading.

Every other pass runs the stock layer and final mixer. The draft model hands a
pass that takes the front to `DraftFront.run_pass` once its tokens are
embedded.

Exchange order (TP > 1). The boundary kernel's ranks exchange their partial
outputs through a mailbox with two generations (`..boundary.all_reduce_hc`).
Its protocol needs two things of the calls of one kernel instance, and the
draft layer's instance gets them as follows.

1. Every rank makes the same calls in the same order. The draft graphs are
   captured and replayed alike on every rank, and inside a capture every
   choice that changes the calls reads state that every rank shares: whether
   a pass takes the front (`takes`: the row count, the capture mode and the
   attention metadata's host fields), whether its reads take the read kernels
   (the row count) and which blocks the kernel serves (decided when the
   boundaries are built and compared across ranks there, then checked again
   from the layer's configuration at each capture). The target's instance is
   another instance, with its own mailbox and generation words; the two never
   read each other's.
2. A call reads the generation word only after the instance's previous call
   on this rank completed. The call reads it after its grid-dependency wait,
   which returns once the kernels it follows on the stream completed. So it
   suffices that every kernel between two consecutive calls completes only
   after the kernels it follows completed: one launched with programmatic
   dependent launch (PDL) waits (`griddepcontrol.wait`) before it
   completes, and one launched without it starts after them.

   - Within a pass, from the attention block's call to the MoE block's: the
     read's projections (`hc_down` and `hc_up_mix`, or the stock read's
     kernels above 64 rows), then the MoE layer's kernels on the main stream
     and on the side stream that forks from it and joins it before the call
     (event edges, which order by completion). Each reads what its
     predecessor on its stream wrote. One launched with PDL (this package's
     kernels, the model's Triton kernels, the planned GEMMs, the input
     quantization, FlashInfer's routed-expert kernels) reads it only after
     its wait; one launched without PDL (PyTorch's, cuBLAS's) starts after
     its predecessor completed. This is the chain of the target's layers,
     with the draft layer's FP8 experts.
   - From the MoE block's call of a pass to the attention block's call of the
     next pass: the final mixer's read kernels, the draft head (the row gather
     of the first pass and the head GEMM), then the token choice
     (`TokenChoice`), each reading its predecessor's output. The token
     choice waits before it reads the logits and never triggers its
     dependents early, so every later kernel starts after it completed:
     the token copies, the speculator's input update and metadata kernels, the
     embedding and its all-reduce, the front, the first read and the QSA
     kernels then need nothing. Without the token choice, the stock choice's
     PyTorch kernels likewise start only after their predecessor completed.
     The first pass and the later passes are separate graph launches, which
     start after the stream's earlier work completed.

   The CPU model in
   `tests/models/qwen4_exp/specialized_decode/test_draft_exchange_order.py`
   applies CUDA's rules for PDL to these chains, with each kernel from
   outside this package at the earliest trigger its contract allows and
   every kernel after the token choice never waiting. It reports the defects
   that would break them: a read's up mix without its wait, a token choice
   that triggers its dependents early, a call that reads the generation word
   before its wait. A call that stalls anyway (a rank that skipped or
   reordered a call) traps after 30 s with a report
   (`SublayerBoundary.stall_report`) instead of hanging.

Warm-up. The front compiles and runs its kernels once at the end of the draft
model's weight loading (`finish_loading`). The specialized decode's model
state warms up the target's kernels, but Model Runner V2 builds that state
from the target model alone; the draft model is loaded just before it, so
warming up at the end of the draft's loading still precedes memory profiling
and every CUDA graph capture (the speculator's included).
"""

import time
from typing import TYPE_CHECKING

import torch

from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import is_forward_context_available
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import UnquantizedLinearMethod

from .. import HC_HIDDEN, HC_STREAMS, DecodeConfig
from ..hc.read import MAX_ROWS, hc_project, hc_read, warm_up_hc_read
from ..mtp import front_norm, front_norm_supports
from .boundary import Boundaries, PendingBlock, moe_unsupported
from .decode import HcRead, PureDecodeGate, stock_project, stock_read
from .gemm import PlannedGemms, draft_front_gemm, draft_front_gemm_plans
from .qsa import QsaRows

if TYPE_CHECKING:
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm
    from vllm.model_executor.layers.linear import ColumnParallelLinear

    from ...hyperconnection import GatedResidual
    from ...model import Qwen4ExpDecoderLayer
    from ...mtp import Qwen4ExpMultiTokenPredictor

logger = init_logger(__name__)

PROJECTIONS = ("fc_embedding", "fc_hidden")
# The refusal of a front whose weights the draft model's load_weights has not
# bound yet (finish_loading).
UNBOUND = "projection weights bound by the draft model's load_weights"


class DraftFront:
    """The fused front of the MTP draft passes (module docstring).

    Args:
        predictor: The MTP draft model being built, with its norms,
            projections and layers.
        config: The deployment's specialized decode settings.

    """

    # The draft layers' sublayer boundaries, built at the end of loading; a
    # class default, so a front built without __init__ runs the stock layer.
    boundaries: Boundaries | None = None

    def __init__(
        self, predictor: "Qwen4ExpMultiTokenPredictor", config: DecodeConfig
    ) -> None:
        # Imported here: the qsa module needs CUDA-only kernels.
        from ...qsa import Qwen4ExpQSAAttention

        self.config = config
        self.layers: tuple[Qwen4ExpDecoderLayer, ...] = tuple(predictor.layers)
        self.final_mixer: GatedResidual = predictor.hyper_connection_mixer
        # The rows of the largest pass whose reads take the read kernels.
        self.kernel_read_rows = min(config.max_rows, MAX_ROWS)
        self.embed_norm: GemmaRMSNorm = predictor.pre_fc_norm_embedding
        self.hidden_norm: GemmaRMSNorm = predictor.pre_fc_norm_hidden
        self.projections: dict[str, ColumnParallelLinear] = {
            name: getattr(predictor, name) for name in PROJECTIONS
        }
        # Why the front cannot serve this deployment, or None.
        refusal = self._unsupported(predictor)
        # The full projection weights the front multiplies by, once bound.
        self.weights: dict[str, torch.Tensor] = {}
        self.replicas: dict[str, torch.Tensor] = {}
        self.loaded: set[str] = set()
        self.gemm_plans = draft_front_gemm_plans()
        # The draft layers' projections in the passes that take the front.
        self.layer_gemms = PlannedGemms.build(predictor.layers)
        # The draft layers' QSA attention in the large passes that take it.
        self.layer_qsa = QsaRows.build(predictor.layers)
        prefill_counting: list[str] = []
        qsa_main: list[str] = []
        for layer in predictor.layers:
            attention = getattr(layer, "self_attn", None)
            if (
                not isinstance(attention, Qwen4ExpQSAAttention)
                or attention.indexer is None
            ):
                refusal = refusal or "QSA attention with an indexer"
                continue
            prefill_counting.append(attention.indexer.raw_key_cache.prefix)
            prefill_counting.append(attention.indexer.compressed_key_cache.prefix)
            qsa_main.append(attention.layer_name)
        self.gate = PureDecodeGate(
            prefill_counting_layers=tuple(prefill_counting),
            qsa_main_layers=tuple(qsa_main),
            max_decode_query_len=config.tokens_per_request,
        )
        if refusal is None and get_tensor_model_parallel_world_size() > 1:
            for name, layer in self.projections.items():
                self._replicate(name, layer)
        # Why every pass runs stock, or None. A deployment the front serves
        # runs stock too until finish_loading binds the weights.
        self.refusal: str | None = refusal or UNBOUND

    def _unsupported(self, predictor: "Qwen4ExpMultiTokenPredictor") -> str | None:
        """Why the front cannot run this draft model's front, or None."""
        if (predictor.hc_count, predictor.hidden_size) != (HC_STREAMS, HC_HIDDEN):
            return f"{HC_STREAMS} hyper-connection streams of {HC_HIDDEN} values"
        for norm in (self.embed_norm, self.hidden_norm):
            if norm.weight.dtype != torch.bfloat16:
                return "BF16 pre-projection norm weights"
        for layer in self.projections.values():
            if not isinstance(layer.quant_method, UnquantizedLinearMethod) or (
                layer.weight.dtype != torch.bfloat16
            ):
                return "unquantized BF16 fc_embedding and fc_hidden"
        return None

    def _replicate(self, name: str, layer: "ColumnParallelLinear") -> None:
        """Allocate the full copy of `layer`'s weight and wrap its loader
        to fill the copy from the whole checkpoint tensor."""
        shard = layer.weight
        self.replicas[name] = torch.empty(
            (HC_HIDDEN, HC_HIDDEN), dtype=shard.dtype, device=shard.device
        )
        stock_loader = shard.weight_loader

        def load(
            param: torch.nn.Parameter, loaded_weight: torch.Tensor, *args, **kwargs
        ):
            """Copy the whole checkpoint tensor, then load the rank's shard."""
            # Looked up per call: a refused front frees its replicas.
            replica = self.replicas.get(name)
            if replica is not None and loaded_weight.shape == replica.shape:
                replica.copy_(loaded_weight)
                self.loaded.add(name)
            return stock_loader(param, loaded_weight, *args, **kwargs)

        shard.weight_loader = load

    def finish_loading(self) -> None:
        """Bind the projection weights once the draft model's weights are
        loaded, refuse (and free the replicas) if a replica was not loaded,
        and warm up the kernels. The draft model calls it at the end of its
        `load_weights`."""
        if self.refusal == UNBOUND:
            missing = [
                name
                for name in PROJECTIONS
                if self.replicas and name not in self.loaded
            ]
            if missing:
                self.refusal = "full checkpoint tensors of " + " and ".join(missing)
            else:
                self.refusal = None
                self.weights = dict(self.replicas) or {
                    name: layer.weight for name, layer in self.projections.items()
                }
        if self.refusal is not None:
            self.weights, self.replicas = {}, {}
            self._keep_stock()
            return
        # Collective over the TP group: every rank maps the others' mailboxes.
        # Every rank gets here or refuses alike (the refusals read the
        # deployment and the checkpoint only).
        # Built before the quantization methods process the weights, so no
        # routed front (it reads the processed expert weights).
        self.boundaries = Boundaries.build(
            self.layers, self.config.max_rows, "the MTP draft layer", routed_front=False
        )
        self.warm_up()

    def _keep_stock(self) -> None:
        """Log, once, why every draft pass runs the stock front."""
        logger.info_once(
            "Qwen4Exp specialized decode: the MTP draft passes keep the stock "
            "front, which needs %s.",
            self.refusal or UNBOUND,
        )

    def takes(self, embedded: torch.Tensor, hidden: torch.Tensor) -> bool:
        """Whether this forward runs the fused front: a FULL CUDA graph capture
        of a pure decode step of 1 to `max_rows` rows, activations the kernel
        takes, and a deployment the front serves with bound weights. Reads
        host state only."""
        rows = embedded.shape[0]
        if (
            not 0 < rows <= self.config.max_rows
            or not is_forward_context_available()
            # Eager forwards, profiling and the eager run before each capture
            # run stock (a replayed graph does not call forward).
            or not torch.cuda.is_current_stream_capturing()
            or not self.gate.captures_pure_decode()
        ):
            return False
        if self.refusal is not None or not self.weights:
            # Refused at construction or at the end of loading, or never
            # bound: a loader skipped the draft model's load_weights.
            self._keep_stock()
            return False
        if not front_norm_supports(embedded, hidden):
            logger.warning_once(
                "Qwen4Exp specialized decode: an MTP draft pass of %d rows keeps "
                "the stock front (activation layout %s, %s).",
                rows,
                tuple(embedded.stride()),
                tuple(hidden.stride()),
            )
            return False
        return True

    def run(
        self, embedded: torch.Tensor, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The front of a pass `takes` accepted: the projected embedding
        `[rows, 2560]` and the projected streams `[rows, 10240]`, as the
        stock front returns them."""
        rows = embedded.shape[0]
        normed_embedded, normed_hidden = front_norm(
            embedded,
            hidden,
            self.embed_norm.weight,
            self.hidden_norm.weight,
            self.embed_norm.variance_epsilon,
            self.hidden_norm.variance_epsilon,
        )
        projected = draft_front_gemm(
            normed_embedded, self.weights["fc_embedding"], self.gemm_plans
        )
        # The four streams of a row are four GEMM rows (the same bytes).
        streams = draft_front_gemm(
            normed_hidden.view(rows * HC_STREAMS, HC_HIDDEN),
            self.weights["fc_hidden"],
            self.gemm_plans,
        )
        return projected, streams.view(rows, HC_STREAMS * HC_HIDDEN)

    def run_pass(
        self,
        predictor: "Qwen4ExpMultiTokenPredictor",
        embedded: torch.Tensor,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        spec_step_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """A draft pass that `takes` accepted, from its embedded tokens on:
        the front (`run`), then the draft layer with its GEMM plans and QSA
        attention (`QsaRows`: no fill of the pass's rows, the row attention
        in passes of `MIN_CTAS` CTAs and more) and
        the final mixer, as `Qwen4ExpMultiTokenPredictor.forward` runs them on
        the last pipeline rank without sequence parallelism (the deployment
        check rejects both). Returns what that forward returns: the single
        stream for the LM head and the multi-stream state of the next pass.

        Args:
            predictor: The MTP draft model whose pass this is.
            embedded: The pass's token embeddings `[rows, 2560]`.
            hidden: The previous pass's streams `[rows, 10240]`.
            positions: The pass's positions.
            spec_step_idx: The draft step, which picks the draft layer.

        """
        projected, streams = self.run(embedded, hidden)
        position = spec_step_idx % predictor.num_mtp_layers
        layer = predictor.layers[position]
        final_mixer = predictor.hyper_connection_mixer
        with self.layer_gemms.active(), self.layer_qsa.active(embedded.shape[0]):
            if self.boundaries is not None:
                return self._run_layer(
                    self.boundaries,
                    position,
                    layer,
                    final_mixer,
                    projected,
                    streams,
                    positions,
                )
            hidden_states, block_output, injection = layer(
                hidden_states=streams,
                prev_block_output=projected,
                prev_injection=None,
                positions=positions,
                input_ids=None,
                query_start_loc=None,
                ngram_context=None,
            )
        multi_hidden, sample_hidden_states, _ = final_mixer.combine_and_mix(
            hidden_states, block_output, injection
        )
        return sample_hidden_states, multi_hidden

    def _run_layer(
        self,
        boundaries: Boundaries,
        position: int,
        layer: "Qwen4ExpDecoderLayer",
        final_mixer: "GatedResidual",
        projected: torch.Tensor,
        streams: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The draft layer at `position` and the final mixer as the
        executor runs a target layer (`SpecializedDecode._run_layers` for a
        QSA layer without PLE or deepstack): the read kernels, the boundary
        kernel and the stock submodule calls. Returns the single stream for
        the LM head and the multi-stream state of the next pass."""
        entry = boundaries.moe_layers[position]
        if entry is not None and (reason := moe_unsupported(entry[0])):
            # The boundaries are built at the end of the draft model's
            # load_weights, before its quantization methods process the
            # weights: a MoE layer that no longer passes would be reduced
            # twice.
            raise RuntimeError(
                "Qwen4Exp specialized decode: the MTP draft layer's MoE block "
                f"left to the boundary kernel no longer has {reason}."
            )
        kernel_reads = projected.shape[0] <= self.kernel_read_rows
        read: HcRead = hc_read if kernel_reads else stock_read
        project = hc_project if kernel_reads else stock_project

        def read_block(
            hc: "GatedResidual",
            hidden_states: torch.Tensor,
            block_output: torch.Tensor | PendingBlock,
            injection: torch.Tensor | None,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
            """The read of `hc` after `block_output`: the boundary kernel
            for a pending block, `read` otherwise."""
            if isinstance(block_output, PendingBlock):
                return boundaries.read(
                    hc, hidden_states, block_output, injection, project
                )
            return read(hc, hidden_states, block_output, injection)

        # The projected embedding joins every stream with unit weight (no
        # injection logits), as the stock layer's first combine.
        hidden_states, block_input, injection = read(
            layer.attn_hyper_connection, streams, projected, None
        )
        with boundaries.attention_partial(layer.self_attn.o_proj) as partial:
            output = layer.self_attn(hidden_states=block_input, positions=positions)
        attn_output = PendingBlock(output) if partial else output
        hidden_states, block_input, injection = read_block(
            layer.mlp_hyper_connection, hidden_states, attn_output, injection
        )
        pending = boundaries.moe_output(position, block_input, True)
        block_output = layer.mlp(block_input) if pending is None else pending
        multi_hidden, sample_hidden_states, _ = read_block(
            final_mixer, hidden_states, block_output, injection
        )
        return sample_hidden_states, multi_hidden

    def warm_up(self) -> None:
        """Compile and run the norm kernel, every planned GEMM of the front and
        of the draft layers and the draft layers' QSA row attention once, so
        that no CUDA graph capture compiles one."""
        start = time.perf_counter()
        device = self.embed_norm.weight.device
        with torch.inference_mode():
            # The norm kernel takes the row count at launch: one compile.
            self.run(
                torch.zeros(1, HC_HIDDEN, dtype=torch.bfloat16, device=device),
                torch.zeros(
                    1, HC_STREAMS * HC_HIDDEN, dtype=torch.bfloat16, device=device
                ),
            )
            # One skinny GEMM build per planned row count.
            for rows in sorted(self.gemm_plans):
                draft_front_gemm(
                    torch.zeros(rows, HC_HIDDEN, dtype=torch.bfloat16, device=device),
                    self.weights["fc_embedding"],
                    self.gemm_plans,
                )
            torch.accelerator.synchronize(device)
        layer_plans = self.layer_gemms.warm_up()
        self.layer_qsa.warm_up()
        self._warm_up_layer_reads()
        logger.info_once(
            "Warmed up the Qwen4Exp specialized decode's MTP draft front (fused "
            "pre-projection norms, %s projections) for 1-%d rows and %d GEMM "
            "plans, the reads and the boundary kernel of the draft layers in "
            "%.1f s.",
            "replicated" if self.replicas else "local",
            self.config.max_rows,
            len(layer_plans),
            time.perf_counter() - start,
        )

    def _warm_up_layer_reads(self) -> None:
        """Run the draft layers' read kernels and boundary kernel variants
        once (the latter collective over the TP group): every row group of
        the read kernels, the combine-norm of the first read without
        injection logits, and the boundary kernel's attention and MoE
        variants."""
        boundaries = self.boundaries
        if boundaries is None or not self.layers:
            return
        layer = self.layers[0]
        final_mixer = self.final_mixer
        weight = layer.attn_hyper_connection.hc_norm.weight
        with torch.inference_mode():
            warm_up_hc_read(
                layer.mlp_hyper_connection, final_mixer, self.kernel_read_rows
            )
            hc_read(
                layer.attn_hyper_connection,
                weight.new_zeros(1, HC_STREAMS * HC_HIDDEN),
                weight.new_zeros(1, HC_HIDDEN),
                None,
            )
            boundaries.warm_up(layer.mlp_hyper_connection)

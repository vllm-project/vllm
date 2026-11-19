# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The GDN (gated delta net) layers of the specialized decode executor.

Stock vLLM runs a Qwen4Exp GDN layer of an MTP decode step
(`QwenGatedDeltaNetAttention.forward_cuda` with the fused spec-decode path)
as the `qkvz` and `ba` input projections, a zero fill of the core output,
the Triton `causal_conv1d_update`, the CUDA
`fused_gdn_decode_post_conv_mtp` and the output projection. `GdnLayers`
runs the layer's own `in_proj_qkvz`, `in_proj_ba` and `out_proj` around
one kernel instead, `gdn_decode_core` (`..gdn`), which computes the same
recurrent states and convolution states bit for bit and the outputs within
the last bit (the core's docstring, Numerics). The `ba`
projection runs on a side stream, forked before the `qkvz` projection and
joined before the core, so that the core follows the `qkvz` projection on
the main stream and launches early (programmatic dependent launch): it loads
its recurrent state while the projection runs. Once the model state binds a
`GdnRecords` (`bind_records`), the core runs the record/commit state
protocol: it writes no recurrent state but per-row records (into the
requests' other speculative slots), and the model state commits the accepted
rows after sampling (`gdn_records.py`).

Staged projections. In tensor-parallel steps of at most 16 rows (TP 4 at one
to four requests), the layer's projections run the staged-weight kernel
`gdn_projection` (`..gdn.proj`) instead: one launch for `qkvz` and
`ba` together, so neither the side stream nor its join remain, and one for
the output projection, which returns this rank's partial output for the
sublayer boundary (`Boundaries.attention_partial`). Each launch copies its
CTA's whole weight share into shared memory before its grid-dependency wait
and triggers its dependents right after issuing the copies, so the weights
stream while the previous kernels run, and the core and the boundary kernel
launch early behind them. The projections round as an FP32-accumulated GEMM
does but sum `k` in another order than the GEMMs they replace, so outputs
are within BF16 rounding of the stock path, not bitwise equal. See
`staged_projection_unsupported` for the conditions; every one is build-time
state or the step's row count, fixed per captured graph and equal on every
rank, so the ranks always take the same path.

Which steps. The executor only runs FULL CUDA graph captures of pure decode
steps. Among them, a GDN layer takes the fused core when its metadata
(`GDNAttentionMetadata`, host fields only, which every TP rank shares) is a
speculative decode of `num_speculative_tokens + 1` rows per request without
prefills or plain decodes, for at most the executor's largest step;
otherwise it runs the stock layer. The whole deployment keeps the stock layer
when the fused core does not reproduce its stock path
(`fused_core_unsupported`): a BF16 recurrent state
(`--mamba-ssm-cache-dtype bfloat16`, a supported secondary configuration
whose stock path stays as it is), more rows per request than the core takes
(`MAX_CORE_TOKENS`, 8; above the stock fused MTP path's limit stock runs the
Triton core, whose rounding differs), or any layer shape or setting the kernel
is not built for.

Stream order. For a fused layer, `GdnLayers` enqueues on the step's stream
M (the capture stream) and its own side stream S, in this order:

1. M records `start` (the fork);
2. S waits for `start`, runs the `ba` projection and records `done`;
3. M runs the `qkvz` projection;
4. M waits for `done` (the join);
5. M launches the core (programmatic dependent launch), then `out_proj`.

What makes this correct:

- The `ba` projection reads the layer's input only after it is written: S
  waits for `start`, which M records after every operation that produces the
  input (the previous layer, its output projection and the hyper-connection
  steps between layers). Every layer's `ba` waits for that layer's own fork,
  so side-stream work never overtakes the main stream from one layer to the
  next. In particular a `ba` output block that the allocator hands to the
  next layer's `ba` on S is rewritten only after M has passed this layer's
  core and output projection.
- The core reads `ba` only after it is written: M waits for `done` before
  it launches the core, so the core's graph node depends on the `ba`
  projection's last node. Should the capture make that edge programmatic too,
  the core still reads `ba` only after its `griddepcontrol.wait`, which
  waits for every prerequisite grid to complete.
- S rejoins M in every fused layer, so a capture never ends with S unjoined.
- Every fused layer's core writes the same output buffer (below). The next
  layer's core writes it only after its own wait. That wait returns after the
  next `qkvz` projection completes, and so after this layer's output
  projection completes, because every kernel on M waits for its predecessor
  before it completes.
- What the core reads before its wait, and what its early dependents may do,
  is in `..gdn.core`.

The executor runs only during CUDA graph captures. There `GdnLayers` checks
the first two points on the captured graph itself, from the streams' capture
dependencies (`cudaStreamGetCaptureInfo`):

- the `ba` projection depends on M's operations at the fork;
- the core depends on the `ba` projection's last node.

A violation raises during the capture. The checks are host calls at capture
time only, so they cost graph replays nothing.

Order of the staged layer. Everything runs on M, each kernel launched with
programmatic dependent launch (PDL):

1. P_in, `gdn_projection(x, [W_qkvz, W_ba])`: reads the layer input `x`
   (the read's `hc_up_mix` output), writes `qkvz` and `ba`;
2. C, the core: reads `qkvz` and `ba`, writes the core output and the
   layer's caches;
3. P_out, `gdn_projection(core output, [W_out])`: writes the partial output;
4. B, the next read's boundary kernel: reads the partial output.

What makes this correct:

- Who waits on what. Every kernel of M reads what an earlier kernel of the
  step wrote, and writes global memory, only after its
  `griddepcontrol.wait`, and completes only after that wait returned. A
  wait returns once the previous kernel completed, so it returns only after
  every earlier kernel of M completed. P_in reads `x`, C reads `qkvz` and
  `ba`, P_out reads the core output and B the partial output, each after
  its wait.
- What runs before a wait. The projections trigger their dependents before
  their own wait, so C may start while `hc_up_mix` and the kernels before
  it still run, and P_out and B while C, P_in and earlier kernels run. Before
  its wait a projection reads only its weights (loaded parameters, never
  written) into its own shared memory and writes nothing else; C reads only
  the step's metadata, loaded weights and this layer's caches (`..gdn.core`)
  and B only its norm weights (`..boundary.all_reduce_hc`). No other kernel
  of the step writes any of these: the metadata and weights are written before
  the step's graph, and only C itself writes this layer's caches, after its
  reads of them (the core's own order).
- No reuse before the readers finish. `qkvz`, `ba` and the partial output
  are allocated on M, and `x` was; the caching allocator hands a released
  block out again only to an allocation on M made after the release, which
  comes after the reader was enqueued (each tensor is referenced until then).
  The kernel that writes the reused block is enqueued after the reader and
  writes only after its wait, so after the reader completed. The core output
  is the executor's one buffer for every layer: the next layer's core writes
  it only after its wait, after this layer's P_out completed. The staged
  weights live in each CTA's own shared memory; their global copies are
  parameters and never written.

The capture checks, on the captured graph: P_in is one kernel node; C's node
depends only on P_in's node (no side stream joins it and nothing ran between
them); P_out's node depends only on C's node. A violation raises during the
capture. `tests/models/qwen4_exp/specialized_decode/test_gdn_chain_order.py`
checks the order above by simulation on what `GdnLayers` enqueues and
releases over three layers, with the projection kernel's accesses read from
its source, and shows that the simulation reports injected defects.

Padded rows. The core writes the rows of the step's requests
(`spec_query_start_loc`); a FULL graph's padding rows past them are not
written. The core's output is an executor-owned buffer, zero-filled once when
the executor is built, so those rows hold zeros or an earlier step's finite
outputs (stock zero-fills them each step). Only row-wise operations read them,
and they never reach sampled outputs. The decode graph gains no fill node.
"""

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    MAX_FUSED_GDN_MTP_TOKENS,
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

if TYPE_CHECKING:
    from ..gdn import Schedule
    from .gdn_records import GdnRecords

logger = init_logger(__name__)

# The layer shape the fused core is built for: 128-dim heads, three value heads
# per key head, a width-4 convolution.
HEAD_DIM = 128
VALUE_HEADS_PER_KEY_HEAD = 3
CONV_WIDTH = 4
# The most rows per request (num_speculative_tokens + 1) the core is validated
# for; the stock fused MTP path, which it reproduces, takes up to
# MAX_FUSED_GDN_MTP_TOKENS.
MAX_CORE_TOKENS = min(8, MAX_FUSED_GDN_MTP_TOKENS)

# gdn_decode_core's signature: (qkvz, ba, conv_weight, conv_state, state,
# a_log, dt_bias, norm_weight, slots, accepted, cu_seqlens, requests, out, eps,
# schedule, records=, layout=, seq_lens=, block_size=) -> out.
CoreKernel = Callable[..., torch.Tensor]
ScheduleFn = Callable[[int, int], "Schedule"]
# gdn_projection's signature: (x, weights) -> [x @ w.T for w in weights].
ProjectionKernel = Callable[[torch.Tensor, list[torch.Tensor]], list[torch.Tensor]]

# The largest step (token rows) whose projections run the staged-weight kernel.
STAGED_MAX_ROWS = 16
# Function-name parts of the captured kernels the capture checks look for.
PROJECTION_KERNEL = "GdnProjection"
CORE_KERNEL = "GdnDecodeCore"


def staged_projection_unsupported(
    layers: Sequence[QwenGatedDeltaNetAttention], sms: int
) -> str:
    """What the staged-weight projections need that `layers` on a device of
    `sms` SMs miss, or `""`.

    They serve tensor-parallel ranks (at TP 1 the weights do not fit in one
    wave of CTAs) with unquantized BF16 projections without bias, whose
    weights fit one CTA per SM (`projection_unsupported`).
    """
    # Imported here: CuTe DSL kernels (CUDA builds only).
    from ..gdn.proj import projection_unsupported

    if not layers:
        return "GDN layers"
    missing: dict[str, None] = {}
    for attn in layers:
        projections = (attn.in_proj_qkvz, attn.in_proj_ba, attn.out_proj)
        if attn.tp_size == 1:
            missing["tensor parallelism"] = None
        weights = [getattr(p, "weight", None) for p in projections]
        if not all(
            isinstance(w, torch.Tensor)
            and w.dtype == torch.bfloat16
            and w.dim() == 2
            and w.is_contiguous()
            and getattr(p, "bias", None) is None
            for p, w in zip(projections, weights)
        ):
            missing["unquantized BF16 projections without bias"] = None
            continue
        hidden = attn.in_proj_qkvz.weight.shape[1]
        for ns, k in (
            (
                [attn.in_proj_qkvz.weight.shape[0], attn.in_proj_ba.weight.shape[0]],
                hidden,
            ),
            ([attn.out_proj.weight.shape[0]], attn.out_proj.weight.shape[1]),
        ):
            need = projection_unsupported(ns, k, sms)
            if need:
                missing[need] = None
    return "; ".join(missing)


def fused_core_unsupported(
    layers: Sequence[QwenGatedDeltaNetAttention], tokens_per_request: int
) -> str:
    """What the fused core needs that `layers` with `tokens_per_request`
    rows per request miss, or `""`."""
    if not layers:
        return "GDN layers"
    missing: dict[str, None] = {}
    if tokens_per_request > MAX_CORE_TOKENS:
        # Above MAX_FUSED_GDN_MTP_TOKENS the stock layer runs the Triton core,
        # which rounds differently from the fused MTP path the core reproduces.
        missing[
            f"at most {MAX_CORE_TOKENS} rows per request "
            "(num_speculative_tokens + 1), the most the fused core takes, "
            f"not {tokens_per_request}"
        ] = None
    for attn in layers:
        conv_dtype, state_dtype = attn.get_state_dtype()
        norm = attn.norm
        requirements = {
            "an FP32 recurrent state (--mamba-ssm-cache-dtype auto or float32)": (
                state_dtype == torch.float32
            ),
            "a BF16 convolution state": conv_dtype == torch.bfloat16,
            # The fused core reproduces this stock path.
            "the stock fused MTP decode path (VLLM_GDN_DECODE_KERNEL=cuda)": (
                attn.enable_fused_gdn_spec_decode
            ),
            "128-dim heads, 3 value heads per key head and a width-4 SiLU "
            "convolution without bias": (
                attn.head_k_dim == attn.head_v_dim == HEAD_DIM
                and attn.num_v_heads == VALUE_HEADS_PER_KEY_HEAD * attn.num_k_heads
                and attn.num_k_heads % attn.tp_size == 0
                and attn.conv_kernel_size == CONV_WIDTH
                and attn.activation == "silu"
                and attn.conv1d.bias is None
            ),
            "the flat q, k, v, z and b, a projection layouts, TP-sharded": (
                not attn.gqa_interleaved_layout
                and not attn.disable_tp_for_ba_proj
                and attn._fused_gdn_gated_rms_norm is None
            ),
            "a per-head RMSNorm before a sigmoid output gate": (
                norm.activation == "sigmoid"
                and norm.norm_before_gate
                and norm.group_size is None
            ),
            "BF16 dt_bias and norm weight and an FP32 A_log": (
                attn.dt_bias.dtype == torch.bfloat16
                and norm.weight.dtype == torch.bfloat16
                and attn.A_log.dtype == torch.float32
            ),
        }
        missing.update((need, None) for need, ok in requirements.items() if not ok)
    return "; ".join(missing)


def capture_dependencies(stream: torch.cuda.Stream) -> frozenset[int]:
    """The captured graph nodes that the next operation enqueued on the
    capturing `stream` will depend on (`cudaStreamGetCaptureInfo`)."""
    # Imported here: CUDA builds only.
    from cuda.bindings import runtime

    # (error, status, id, graph, dependencies, ...): CUDA 13 also returns the
    # edge data before the count.
    info = runtime.cudaStreamGetCaptureInfo(stream.cuda_stream)
    if info[0] != runtime.cudaError_t.cudaSuccess:
        raise RuntimeError(f"cudaStreamGetCaptureInfo failed: {info[0]}")
    return frozenset(int(node) for node in info[4])


def require_dependencies(
    stream: torch.cuda.Stream, nodes: frozenset[int], what: str, on: str
) -> None:
    """Raise unless the next operation captured on `stream` (`what`)
    depends on every one of `nodes` (`on`)."""
    if not nodes <= capture_dependencies(stream):
        raise RuntimeError(
            f"Qwen4Exp specialized decode: in the captured GDN layer, {what} "
            f"does not wait for {on}."
        )


def captured_kernel(
    stream: torch.cuda.Stream, name: str, after: int | None, what: str, on: str
) -> int:
    """The graph node just captured on `stream` (`what`). Raises unless it
    is one kernel node whose function name contains `name` and, when
    `after` is given, whose only dependency is that node (`on`)."""
    # Imported here: boundary imports this module.
    from .boundary import kernel_name, node_dependencies

    nodes = capture_dependencies(stream)
    node = min(nodes) if len(nodes) == 1 else None
    if node is None or name not in kernel_name(node):
        raise RuntimeError(
            f"Qwen4Exp specialized decode: in the captured GDN layer, {what} is "
            "not the one kernel just captured on the stream."
        )
    if after is not None and node_dependencies(node) != frozenset((after,)):
        raise RuntimeError(
            f"Qwen4Exp specialized decode: in the captured GDN layer, {what} "
            f"does not follow {on} alone."
        )
    return node


class GdnLayers:
    """Runs the GDN layers of the executor's steps: the fused core where it
    applies, the stock layer otherwise.

    Args:
        layers: The backbone's local GDN layers.
        tokens_per_request: Rows of one request (the MTP draft tokens plus one).
        max_requests: Requests of the largest step the executor runs.
        core: The fused core (`gdn_decode_core`), or None to run every
            layer stock.
        schedule: The core's schedule for (key heads, requests)
            (`default_schedule`).
        stream: The side stream of the `ba` projection (None runs it after
            the `qkvz` projection on the current stream).
        events: The fork and join events (`start`, `done`); two new CUDA
            events by default when `stream` is given.
        projection: The staged-weight projections (`gdn_projection`) for
            steps of up to `STAGED_MAX_ROWS` rows, or None to run the
            layer's own projections in every step.

    """

    def __init__(
        self,
        layers: Sequence[QwenGatedDeltaNetAttention],
        tokens_per_request: int,
        max_requests: int,
        core: CoreKernel | None,
        schedule: ScheduleFn | None,
        stream: torch.cuda.Stream | None,
        events: tuple[torch.cuda.Event, torch.cuda.Event] | None = None,
        projection: ProjectionKernel | None = None,
    ) -> None:
        self.layers = list(layers)
        self.tokens_per_request = tokens_per_request
        self.max_requests = max_requests
        self.core = core
        assert projection is None or core is not None
        self.projection = projection
        self.schedules: dict[int, Schedule] = {}
        self.key_heads = 0
        # The core's output rows: zero-filled once, see the module docstring.
        self.out: torch.Tensor | None = None
        if core is not None:
            assert schedule is not None and self.layers
            first = self.layers[0]
            self.key_heads = first.num_k_heads // first.tp_size
            self.schedules = {
                requests: schedule(self.key_heads, requests)
                for requests in range(1, max_requests + 1)
            }
            self.out = torch.zeros(
                max_requests * tokens_per_request,
                first.value_dim // first.tp_size,
                dtype=torch.bfloat16,
                device=first.A_log.device,
            )
        self.stream = stream
        if events is None and stream is not None:
            events = (torch.cuda.Event(), torch.cuda.Event())
        self.events = events
        # The record/commit state protocol (gdn_records.py), once bound.
        self.records: GdnRecords | None = None

    def bind_records(self, records: "GdnRecords") -> None:
        """Run the fused core with the `records` state protocol of
        `records` (`gdn_records.py`) from now on."""
        assert self.core is not None
        self.records = records

    @classmethod
    def build(
        cls,
        layers: Sequence[QwenGatedDeltaNetAttention],
        tokens_per_request: int,
        max_requests: int,
    ) -> "GdnLayers":
        """The GDN layers of an executor: with the fused core when it covers
        `layers` (logging why not otherwise)."""
        missing = fused_core_unsupported(layers, tokens_per_request)
        if missing:
            logger.info_once(
                "Qwen4Exp specialized decode: the GDN layers run the stock path "
                "(the fused GDN decode core needs %s).",
                missing,
            )
            return cls(layers, tokens_per_request, max_requests, None, None, None)
        # Imported here: CuTe DSL kernels (CUDA builds only).
        from ..gdn import default_schedule, gdn_decode_core
        from ..gdn.proj import gdn_projection

        device = layers[0].A_log.device
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        staged_missing = staged_projection_unsupported(layers, sms)
        if staged_missing:
            logger.info_once(
                "Qwen4Exp specialized decode: the GDN layers run their own "
                "projections (the staged-weight projections need %s).",
                staged_missing,
            )
        return cls(
            layers,
            tokens_per_request,
            max_requests,
            gdn_decode_core,
            default_schedule,
            torch.cuda.Stream(),
            projection=None if staged_missing else gdn_projection,
        )

    def staged(self, rows: int) -> bool:
        """Whether a step of `rows` rows runs the staged-weight projections
        (fixed per captured graph and equal on every rank)."""
        return self.projection is not None and rows <= STAGED_MAX_ROWS

    def warm_up(self) -> list["Schedule"]:
        """Compile and load the fused core for every schedule the executor's
        steps can use, and the staged-weight projections when the layers use
        them; returns the schedules (none without the fused core)."""
        if self.core is None:
            return []
        # Imported here, as in build: CuTe DSL kernels.
        from ..gdn import warm_up_gdn_decode_core

        schedules = list(dict.fromkeys(self.schedules.values()))
        warm_up_gdn_decode_core(
            self.key_heads,
            self.tokens_per_request,
            schedules,
            self.layers[0].A_log.device,
            None if self.records is None else self.records.layout,
            self.records is not None and self.records.align,
        )
        if self.projection is not None:
            from ..gdn.proj import warm_up_gdn_projection

            first = self.layers[0]
            warm_up_gdn_projection(
                [
                    [first.in_proj_qkvz.weight, first.in_proj_ba.weight],
                    [first.out_proj.weight],
                ]
            )
        return schedules

    def uses_core(self, metadata: AttentionMetadata | None) -> bool:
        """Whether a layer with step `metadata` runs the fused core: a
        speculative decode of `tokens_per_request` rows per request without
        prefills or plain decodes, of at most `max_requests` requests."""
        return (
            self.core is not None
            and isinstance(metadata, GDNAttentionMetadata)
            and metadata.num_prefills == 0
            and metadata.num_decodes == 0
            and 0 < metadata.num_spec_decodes <= self.max_requests
            and metadata.spec_state_indices_tensor is not None
            and metadata.spec_state_indices_tensor.size(1) == self.tokens_per_request
            and metadata.spec_query_start_loc is not None
            and metadata.num_accepted_tokens is not None
        )

    def project(
        self, attn: QwenGatedDeltaNetAttention, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, frozenset[int] | None]:
        """The layer's `qkvz` projection on the current stream and its `ba`
        projection on the side stream, forked and joined as the module
        docstring orders them. Returns both and, during a capture, the `ba`
        projection's last graph nodes (None otherwise)."""
        if self.stream is None or self.events is None:
            return (
                attn.in_proj_qkvz(hidden_states)[0],
                attn.in_proj_ba(hidden_states)[0],
                None,
            )
        start, done = self.events
        main, side = torch.cuda.current_stream(), self.stream
        capturing = torch.cuda.is_current_stream_capturing()
        start.record(main)
        fork = capture_dependencies(main) if capturing else None
        # ba first on the side stream, as the stock ba overlap orders it.
        with torch.cuda.stream(side):
            side.wait_event(start)
            if fork is not None:
                require_dependencies(side, fork, "the ba projection", "the fork")
            ba, _ = attn.in_proj_ba(hidden_states)
            ba_nodes = capture_dependencies(side) if capturing else None
            done.record(side)
        mixed_qkvz, _ = attn.in_proj_qkvz(hidden_states)
        main.wait_event(done)
        return mixed_qkvz, ba, ba_nodes

    def __call__(
        self,
        attn: QwenGatedDeltaNetAttention,
        hidden_states: torch.Tensor,
        metadata: AttentionMetadata | None,
    ) -> torch.Tensor:
        """The GDN layer `attn` on its input `hidden_states`, with the
        step's metadata of the layer; returns the output projection."""
        rows = hidden_states.size(0)
        if not self.uses_core(metadata) or self.out is None or rows > self.out.size(0):
            return attn(hidden_states=hidden_states)
        assert self.core is not None and isinstance(metadata, GDNAttentionMetadata)
        staged = self.staged(rows)
        main: torch.cuda.Stream | None = None
        projection_node: int | None = None
        if staged:
            assert self.projection is not None
            mixed_qkvz, ba = self.projection(
                hidden_states, [attn.in_proj_qkvz.weight, attn.in_proj_ba.weight]
            )
            if torch.cuda.is_current_stream_capturing():
                main = torch.cuda.current_stream()
                projection_node = captured_kernel(
                    main, PROJECTION_KERNEL, None, "the qkvz and ba projection", ""
                )
        else:
            mixed_qkvz, ba, ba_nodes = self.project(attn, hidden_states)
            if ba_nodes is not None:
                require_dependencies(
                    torch.cuda.current_stream(),
                    ba_nodes,
                    "the core",
                    "the ba projection",
                )
        requests = metadata.num_spec_decodes
        conv_state = attn.kv_cache[0]
        if not is_conv_state_dim_first():
            conv_state = conv_state.transpose(-1, -2)
        weight = attn.conv1d.weight
        core_out = self.out[:rows]
        records = self.records
        protocol: dict[str, object] = {}
        if records is not None:
            protocol = {"records": records.header(attn), "layout": records.layout}
            if records.align:
                # The copies avoid the boundary row's slot (gdn_records.py,
                # invariant 6), from the runner's seq_lens.
                if records.seq_lens is None:
                    raise RuntimeError(
                        "Qwen4Exp specialized decode: the GDN records core in "
                        "align mode runs before the runner's seq_lens is bound."
                    )
                protocol.update(
                    seq_lens=records.seq_lens, block_size=records.block_size
                )
        self.core(
            mixed_qkvz,
            ba,
            weight.view(weight.size(0), weight.size(2)),
            conv_state,
            attn.kv_cache[1],
            attn.A_log,
            attn.dt_bias,
            attn.norm.weight,
            metadata.spec_state_indices_tensor,
            metadata.num_accepted_tokens,
            metadata.spec_query_start_loc,
            requests,
            core_out,
            attn.layer_norm_epsilon,
            self.schedules[requests],
            **protocol,
        )
        if records is not None:
            records.launched(attn)
        core_node: int | None = None
        if projection_node is not None:
            assert main is not None
            core_node = captured_kernel(
                main, CORE_KERNEL, projection_node, "the core", "the projection"
            )
        if staged and not attn.out_proj.reduce_results:
            # This rank's partial output, which the next read's boundary
            # kernel reduces (Boundaries.attention_partial).
            assert self.projection is not None
            (output,) = self.projection(core_out, [attn.out_proj.weight])
            if core_node is not None:
                assert main is not None
                captured_kernel(
                    main,
                    PROJECTION_KERNEL,
                    core_node,
                    "the output projection",
                    "the core",
                )
            return output
        output, _ = attn.out_proj(core_out)
        return output

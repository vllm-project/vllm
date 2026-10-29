# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The provider surface of the specialized decode and the seams between providers.

One decode step is: the stock embedding, then per layer the input boundary
(gather + dual norm into the attention), the attention (QKV front, attention
core, o_proj partial), the attention boundary (all-reduce + dual norm into the
MoE) and the MoE (router, routing, pooler, routed experts, tail, unpooler), then
the final boundary (the last boundary with the final norm). Each block after the
embedding is implemented by a provider; ``LAYOUTS`` picks the provider of every
block per TP size.

Shapes below use b rows (b <= PREPARED_MAX), H = HIDDEN and P = POOLED. Every
provider launches on the current stream; one that forks a side stream joins it
before ``run`` returns.
"""

from typing import TYPE_CHECKING, Any, NamedTuple, Protocol, TypeAlias

import torch
from torch import nn

if TYPE_CHECKING:
    from ..cutedsl.boundary.all_gather_dual_norm import MailboxSlot

# Decode batches (tokens) the executor serves: one request verifying 1 + 7 tokens.
BATCHES = (8,)
# Rows every executor buffer and provider scratch is sized for.
PREPARED_MAX = 16
HIDDEN = 8192
POOLED = 4096
NUM_EXPERTS = 256
TOP_K = 8


class Step(NamedTuple):
    """A planned step (host state only, the same on every rank)."""

    # Tokens per request.
    q: int
    # [b] int64, the step's positions.
    positions: torch.Tensor
    # Per layer: that layer's attention metadata.
    metadata: tuple[Any, ...]
    # The forward context's slot mapping (per attention layer name).
    slot_mapping: Any


class BuildContext(NamedTuple):
    """What ``Provider.build`` receives, once, after the weights are loaded."""

    # The stock MuseSpark1xModel; providers read it and never modify it.
    backbone: nn.Module
    tp: int
    rank: int
    device: torch.device
    # The TP group's torch.distributed process group.
    group: Any
    # Executor-owned buffers by name, sized for PREPARED_MAX rows.
    buffers: dict[str, torch.Tensor]
    # The layers whose input stream the drafter reads (the stock aux states).
    aux_layers: tuple[int, ...]
    # The deployment's max_model_len.
    max_context: int
    # The sum order of the stock TP all-reduce, which the fused all-reduces
    # take (``policy.FUSED_TPS``; ``flashinfer_ops.stock_sum_order``), else None.
    all_reduce_order: str | None


class Provider(Protocol):
    """``build`` runs once after load, before KV binding and CUDA graph capture:
    every rendezvous, weight view, compile, warm-up and constant happens there.
    ``run`` is graph-capturable, allocates nothing and takes no per-step constant.
    """

    @classmethod
    def unsupported(cls, vllm_config) -> str | None:
        """A deployment requirement of this provider that fails, or None."""
        ...

    @classmethod
    def build(cls, ctx: BuildContext, layer: nn.Module | None) -> "Provider": ...

    def prepare_kv(self) -> None:
        """Called for every bound KV cache (the profiling cache, then the real one)."""
        ...


class BoundaryProvider(Provider, Protocol):
    """The input and final boundaries: all-gather of the MoE's unpooler column
    blocks from ``slot``, then post_ffn_norm, the gated add into ``stream`` (fp32
    [b, H], in place) and the next pre-norm (the next layer's pre_qkv_proj_norm;
    model.norm for the final boundary) into ``out`` (bf16 [b, H]). ``aux`` (fp32
    [b, H]) receives the updated stream when the next layer is an aux layer."""

    def run(
        self,
        slot: "MailboxSlot",
        stream: torch.Tensor,
        out: torch.Tensor,
        aux: torch.Tensor | None,
    ) -> None: ...


class AttentionProvider(Provider, Protocol):
    """The attention: QKV front (QKV GEMM, QK norm, RoPE, KV insert), the
    attention core with o-norm and sigmoid gate, and o_proj. ``x`` bf16 [b, H]
    in; returns the bf16 [b, H] tensor holding this rank's row-parallel o_proj
    partial (before the all-reduce)."""

    def run(self, step: Step, metadata: Any, x: torch.Tensor) -> torch.Tensor: ...


class AttentionBoundaryProvider(Provider, Protocol):
    """The attention boundary: all-reduce of ``partial`` (bf16 [b, H]),
    post_attn_norm, the gated add into ``stream`` (fp32 [b, H], in place) and
    pre_ffn_norm into ``out`` (bf16 [b, H])."""

    def run(
        self, partial: torch.Tensor, stream: torch.Tensor, out: torch.Tensor
    ) -> None: ...


class MoEProvider(Provider, Protocol):
    """The MoE: router (fp32 GateLinear [b, H] -> [b, NUM_EXPERTS]), routing
    (``RoutedTopk``), pooler GEMM and gather + pre_expert_norm + quant
    (``ExpertsInput``), the routed experts (``RoutedExpertsOps`` ->
    ``UnfinalizedExperts``), the tail (post_expert_norm, weighted sum, EP
    combine) and the unpooler GEMM, whose [b, H / tp] column block is published,
    zero-copy, into the mailbox of the next boundary (bound at build)."""

    def run(self, x: torch.Tensor) -> None: ...


class RoutedTopk(NamedTuple):
    """Exactly the pair the stock layer hands the routed experts op: global
    expert ids and their renormalized, scaled weights, as
    fused_musespark1x_routing returns them."""

    # int32 [b, TOP_K]
    ids: torch.Tensor
    # float32 [b, TOP_K]
    weights: torch.Tensor


class ExpertsInput(NamedTuple):
    """The NVFP4 expert input the stock layer hands the routed experts op."""

    # uint8 [b, P // 2], two e2m1 codes per byte.
    a: torch.Tensor
    # float8_e4m3fn [b, P // 16] block scales, in the layout the stock pooler
    # gather emits.
    a_sf: torch.Tensor


# The BF16 edge layers' expert input, bf16 [b, P] pre_expert_norm rows.
ExpertsInputBf16: TypeAlias = torch.Tensor
# Exactly the unfinalized output the stock routed experts op returns
# (do_finalize False, converted as the stock layer converts it), consumed by the
# tail.
UnfinalizedExperts: TypeAlias = Any


class RoutedExpertsOps(Protocol):
    """The two public routed expert ops the stock layer calls and the TP16
    all-reduce wrapper, behind the one runtime module allowed to name their
    library."""

    def build(self, group: Any, layers: tuple[nn.Module, ...]) -> None:
        """Asserts and one warm-up call per op at b = 8; raises where an op is not
        allowed (all_reduce off TP16)."""
        ...

    def nvfp4_plan(self, layer: int, inputs: ExpertsInput, topk: RoutedTopk) -> Any:
        """Host only: how the NVFP4 call for these inputs runs from routing
        metadata (its ``metadata`` is what the routing writes), or None."""
        ...

    def routed_experts(
        self, layer: int, inputs: ExpertsInput, topk: RoutedTopk, plan: Any = None
    ) -> UnfinalizedExperts: ...

    def routed_experts_bf16(
        self, layer: int, x: ExpertsInputBf16, router_logits: torch.Tensor
    ) -> UnfinalizedExperts:
        """The BF16 op routes inside, from the fp32 [b, NUM_EXPERTS] router
        logits, as the stock BF16 layers do."""
        ...

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor: ...


def _scratch(ctx: BuildContext, name: str, shape, dtype) -> torch.Tensor:
    """An executor buffer shared by every layer (allocated once, at build)."""
    buffer = ctx.buffers.get(name)
    if buffer is None:
        buffer = torch.empty(shape, dtype=dtype, device=ctx.device)
        ctx.buffers[name] = buffer
    assert buffer.shape == shape and buffer.dtype == dtype, name
    return buffer


class QkvAttention:
    """The attention: the QKV front (the CuTe kernel of
    cutedsl/attention/qkv_front.py: QKV + o-gate GEMM, QK norm, RoPE and the
    K/V insert), attn2 (the CuTe kernel of cutedsl/attention/attn2.py: paged
    attention of the 8-token verify step with the o-norm and sigmoid gate) and
    the o_proj partial (``OProjection``: cuBLAS; at TP8 the package GEMM, which
    also publishes the partial into the attention boundary's mailbox)."""

    def __init__(
        self,
        attn: nn.Module,
        oproj: Any,
        qkv: torch.Tensor,
        out: torch.Tensor,
        partial: torch.Tensor,
        max_context: int,
        window: int,
        num_splits: int,
        workspace: torch.Tensor | None,
    ):
        self.attn = attn
        self.layer_name = attn.attn.layer_name
        self.max_context, self.window = max_context, window
        self.num_splits, self.workspace = num_splits, workspace
        self.weight = attn.qkv_proj.weight
        self.cos_sin_cache = attn.rotary_emb.cos_sin_cache if attn.use_rope else None
        self.eps = attn.rms_norm_eps
        self.num_kv_heads = attn.num_kv_heads
        self.oproj, self.qkv, self.out, self.partial = oproj, qkv, out, partial

    @classmethod
    def unsupported(cls, vllm_config) -> str | None:
        from . import cute_ops

        if not cute_ops.qkv_front_available():
            return cute_ops.unavailable(cute_ops.QKV_FRONT_KERNEL)
        if not cute_ops.attn2_available():
            return cute_ops.unavailable(cute_ops.ATTN2_KERNEL)
        return None

    @classmethod
    def build(cls, ctx: BuildContext, layer: nn.Module | None) -> "QkvAttention":
        from ..cutedsl.oproj.batt import OProjection
        from . import cute_ops

        assert layer is not None
        attn = layer.self_attn
        reason = cute_ops.unsupported_attn2(attn) or cute_ops.unsupported_projection(
            attn
        )
        if reason is None and (
            attn.use_attn_temperature_tuning or not attn.fused_qk_norm_rope_kv_insert
        ):
            reason = "attention without the stock fused QK-norm/RoPE/KV insert"
        if reason is not None:
            raise ValueError(f"layer {layer.layer_idx}: {reason}")
        n = attn.qkv_proj.weight.shape[0]
        cute_ops.prepare_qkv_front(n, BATCHES)
        qkv = _scratch(ctx, "qkv", (PREPARED_MAX, n), torch.bfloat16)
        out = _scratch(
            ctx,
            "attention",
            (PREPARED_MAX, attn.num_heads * attn.head_dim),
            torch.bfloat16,
        )
        partial = _scratch(ctx, "partial", (PREPARED_MAX, HIDDEN), torch.bfloat16)
        # attn2 splits a long context over a fixed number of clusters (the
        # deployment's, so the captured launch fits every step); the split
        # merge's workspace is shared by the layers, which run in order, sized
        # for the full-attention layers' splits (a window only lowers them)
        # and zeroed once (every launch leaves its counters zero).
        window = cute_ops.attn2_window(attn.attn.sliding_window, ctx.max_context)
        num_splits = cute_ops.attn2_splits(ctx.max_context, window, attn.num_kv_heads)
        cute_ops.prepare_attn2(ctx.max_context, window, attn.num_kv_heads, num_splits)
        workspace = None
        if cute_ops.attn2_split_kernel(ctx.max_context, window):
            size = cute_ops.attn2_workspace_bytes(
                PREPARED_MAX // cute_ops.QUERY_LEN,
                attn.num_kv_heads,
                cute_ops.attn2_splits(ctx.max_context, 0, attn.num_kv_heads),
            )
            workspace = ctx.buffers.get("attn2_workspace")
            if workspace is None:
                workspace = torch.zeros(size, dtype=torch.uint8, device=ctx.device)
                ctx.buffers["attn2_workspace"] = workspace
            assert workspace.shape == (size,), "attn2_workspace"
        return cls(
            attn,
            OProjection.build(ctx, layer),
            qkv,
            out,
            partial,
            ctx.max_context,
            window,
            num_splits,
            workspace,
        )

    def prepare_kv(self) -> None:
        from . import cute_ops

        kv_cache = self.attn.attn.kv_cache
        if not isinstance(kv_cache, torch.Tensor) or kv_cache.numel() == 0:
            return
        if not cute_ops.kv_cache_layout_ok(kv_cache, self.num_kv_heads):
            raise ValueError(
                f"{self.layer_name}: KV cache {tuple(kv_cache.shape)} "
                f"{kv_cache.stride()} {kv_cache.dtype} is not the 16-byte aligned "
                f"bf16 [pages, {self.num_kv_heads}, 16, 128] layout with dense "
                "pages the decode kernels read"
            )

    def front_call(
        self, step: Step, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """(kv_cache, slot_mapping) of this layer's QKV front call on ``x``
        (the step's attention input) into its qkv rows in the current forward
        context, or None if the kernel does not take the call. Host state
        only: shapes, dtypes, strides and alignment, the same on every rank
        (prepare_kv holds every rank's KV cache to the aligned layout)."""
        from vllm.model_executor.layers.attention.attention import (
            get_attention_context,
        )

        from . import cute_ops

        _, _, kv_cache, slot_mapping = get_attention_context(self.layer_name)
        if slot_mapping is None or not cute_ops.supports_call(
            x,
            self.weight,
            step.positions,
            self.cos_sin_cache,
            slot_mapping,
            kv_cache,
            self.num_kv_heads,
            self.qkv[: x.shape[0]],
        ):
            return None
        return kv_cache, slot_mapping

    def run(self, step: Step, metadata: Any, x: torch.Tensor) -> torch.Tensor:
        from . import cute_ops

        b = x.shape[0]
        # plan() leaves a step the front does not take to the stock forward;
        # front_call is the launch's check.
        call = self.front_call(step, x)
        if call is None:
            raise RuntimeError(
                f"{self.layer_name}: the QKV front cannot take this step"
            )
        kv_cache, slot_mapping = call
        qkv = self.qkv[:b]
        cute_ops.launch_qkv_front(
            x,
            self.weight,
            step.positions,
            self.cos_sin_cache,
            slot_mapping,
            kv_cache,
            qkv,
            self.eps,
        )
        q, gate = cute_ops.split_qkv(qkv, self.num_kv_heads)
        decode = metadata.decode
        out = self.out[:b]
        cute_ops.attn2(
            q,
            kv_cache,
            decode.block_tables,
            decode.seq_lens,
            gate,
            out,
            self.num_kv_heads,
            max_seq_len=self.max_context,
            window=self.window,
            num_splits=self.num_splits,
            workspace=self.workspace,
        )
        return self.oproj.run(out, self.partial[:b])


def router_tier(router: nn.Module) -> str | None:
    """Why the stock GateLinear would not run a vLLM-native kernel for the decode
    batch (its last tier is F.linear, which follows --linear-backend), or None."""
    native = (
        "allow_ll_bf16_gemm",
        "allow_fp32_router_gemm",
        "allow_ll_fp32w_gemm",
        "allow_bf16x3_router_gemm",
    )
    if not any(getattr(router, name, False) for name in native):
        return "a router GEMM outside vLLM's native GateLinear tiers"
    return None


class RoutedMoE:
    """The MoE: the stock router (GateLinear, asserted on a vLLM-native tier), the
    routing short tail (``RoutedTopk``; the BF16 edge layers route inside the
    experts op), the pooler (``ExpertsInput`` or ``ExpertsInputBf16``), the routed
    experts (``RoutedExpertsOps`` -> ``UnfinalizedExperts``), the tail and the
    unpooler GEMM, which publishes this rank's column block zero-copy into the
    next boundary's mailbox.

    With the short tail, the router and the routing run on an aux stream while the
    pooler runs on the current one, joined before the routed experts. The fork and
    join are issued directly, not through ``maybe_execute_in_parallel``, which runs
    both sides in series under this model's breakable CUDA graph capture. The BF16
    edge layers stay in series: their router logits feed the experts op on the
    current stream."""

    def __init__(
        self,
        index: int,
        moe: nn.Module,
        ops: Any,
        pooler: Any,
        pooler_index: int,
        tail: Any,
        unpooler: Any,
        routing: Any,
        buffers: dict[str, torch.Tensor],
        aux_stream: torch.cuda.Stream,
    ):
        self.index, self.moe, self.ops = index, moe, ops
        self.pooler, self.pooler_index = pooler, pooler_index
        self.tail, self.unpooler, self.routing = tail, unpooler, routing
        self.ids, self.weights = buffers["topk_ids"], buffers["topk_weights"]
        self.moe_out, self.columns = buffers["moe_out"], buffers["columns"]
        self.aux_stream = aux_stream
        self.forked, self.routed = torch.cuda.Event(), torch.cuda.Event()

    @classmethod
    def unsupported(cls, vllm_config) -> str | None:
        from . import cute_ops

        types = getattr(vllm_config.model_config.hf_text_config, "mlp_layer_types", ())
        if any(kind != "sparse" for kind in types):
            return "dense MLP layers"
        if not cute_ops.routing_available():
            return cute_ops.unavailable(cute_ops.ROUTING_KERNEL)
        return None

    @classmethod
    def build(
        cls,
        ctx: BuildContext,
        layer: nn.Module,
        *,
        ops: Any,
        pooler: Any,
        pooler_index: int,
        tail: Any,
        mailbox: Any,
    ) -> "RoutedMoE":
        """``mailbox``: the LamportMailbox of the boundary after this layer."""
        from vllm.utils.torch_utils import aux_stream

        from ..cutedsl.gemm.unpooler import Unpooler
        from .cute_ops import MuseRouting

        moe = layer.feed_forward
        if moe.use_mega_moe or moe.pooler is None or moe.unpooler is None:
            raise ValueError(f"layer {layer.layer_idx}: unsupported MoE form")
        reason = router_tier(moe.router)
        if reason is not None:
            raise ValueError(f"layer {layer.layer_idx}: {reason}")
        # The stock layer feeds the router the bf16 MoE input at decode batches
        # (an fp32 copy only above 32 tokens); the router here reads the same.
        if any(moe.wants_fp32_router_input(b) for b in BATCHES):
            raise ValueError(
                f"layer {layer.layer_idx}: an fp32 router input at the decode batch"
            )
        routing = None if ops.is_bf16(layer.layer_idx) else MuseRouting(moe.moe_router)
        buffers = {
            "topk_ids": _scratch(ctx, "topk_ids", (PREPARED_MAX, TOP_K), torch.int32),
            "topk_weights": _scratch(
                ctx, "topk_weights", (PREPARED_MAX, TOP_K), torch.float32
            ),
            "moe_out": _scratch(ctx, "moe_out", (PREPARED_MAX, POOLED), torch.bfloat16),
            "columns": _scratch(
                ctx, "columns", (PREPARED_MAX, HIDDEN // ctx.tp), torch.bfloat16
            ),
        }
        unpooler = Unpooler.from_moe(moe, ctx.tp, ctx.rank, mailbox=mailbox)
        tail_layer = tail.bind(moe.post_expert_norm)
        # The router's kernel for the decode batch, before capture.
        moe.router(
            torch.zeros(BATCHES[0], HIDDEN, dtype=torch.bfloat16, device=ctx.device)
        )
        provider = cls(
            layer.layer_idx,
            moe,
            ops,
            pooler,
            pooler_index,
            tail_layer,
            unpooler,
            routing,
            buffers,
            aux_stream(),
        )
        provider.warm_up()
        return provider

    def warm_up(self) -> None:
        """One eager pass, before capture, of the NVFP4 experts path a captured
        step runs: the plan, the routing, the experts and the tail, so that
        every first launch and first-use compile happens before capture. The
        routes spread over all experts, so every EP rank runs its GEMMs.
        Collective (the tail): every rank warms the same layers in the same
        order."""
        if self.routing is None:
            return
        b = BATCHES[0]
        device = self.ids.device
        inputs = ExpertsInput(
            torch.zeros((b, POOLED // 2), dtype=torch.uint8, device=device),
            torch.zeros((b, POOLED // 16), device=device).to(torch.float8_e4m3fn),
        )
        topk = RoutedTopk(self.ids[:b], self.weights[:b])
        plan = self.ops.nvfp4_plan(self.index, inputs, topk)
        # Sigmoid scores of 1 for b * TOP_K experts evenly apart, 0 elsewhere.
        stride = NUM_EXPERTS // (b * TOP_K)
        chosen = torch.arange(b * TOP_K, device=device).view(b, TOP_K) * stride
        logits = torch.full((b, NUM_EXPERTS), -30.0, device=device)
        logits.scatter_(1, chosen, 30.0)
        self.routing(
            logits, topk.ids, topk.weights, None if plan is None else plan.metadata
        )
        experts = self.ops.routed_experts(self.index, inputs, topk, plan)
        self.tail.run(experts, self.moe_out[:b])

    def prepare_kv(self) -> None:
        pass

    def logits(self, x: torch.Tensor):
        logits = self.moe.router(x)
        return logits[0] if isinstance(logits, tuple) else logits

    def run(self, x: torch.Tensor) -> None:
        b = x.shape[0]
        if self.routing is None:
            logits = self.logits(x)
            rows = self.pooler.run(self.pooler_index, x)
            experts = self.ops.routed_experts_bf16(self.index, rows, logits)
        else:
            self.forked.record()
            a, a_sf = self.pooler.run(self.pooler_index, x)
            inputs = ExpertsInput(a, a_sf)
            topk = RoutedTopk(self.ids[:b], self.weights[:b])
            # With a plan the routing also writes the experts' routing metadata,
            # and the experts op runs without a routing kernel of its own.
            plan = self.ops.nvfp4_plan(self.index, inputs, topk)
            with torch.cuda.stream(self.aux_stream):
                self.forked.wait()
                self.routing(
                    self.logits(x),
                    topk.ids,
                    topk.weights,
                    None if plan is None else plan.metadata,
                )
                self.routed.record()
            self.routed.wait()
            experts = self.ops.routed_experts(self.index, inputs, topk, plan)
        moe_out = self.moe_out[:b]
        self.tail.run(experts, moe_out)
        self.unpooler.run(moe_out, self.columns[:b])

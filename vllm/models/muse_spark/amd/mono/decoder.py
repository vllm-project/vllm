# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MuseSpark1x decode monokernel: layers 1-60 of a verify step as FlyDSL launches.

Per eligible step (1-16 requests x 8 verify tokens, TP8 with expert
parallelism; see README.md):

- layer 0: stock.
- layer 1: stock attention input and attention front (QKV, QK-norm, RoPE, KV
  insert), then vLLM's attention op.
- layers 1-60: from layer i's attention output to layer i+1's q, o-gate and
  KV insert: o-norm/gate, o_proj, the attention boundary, the MoE with its EP
  all-reduce, the FFN boundary, layer i+1's QKV+O-gate GEMV, QK-norm and RoPE.
  8 / 16 rows: one K3 launch; 24-128 rows: HA, HM (the experts) and HB.
  vLLM's attention op on that q is layer i+1's attention.
- layer 61: vLLM's attention op, then the stock tail.
- fused attention (8, 16 and 24-32 rows): layer i's attention (i > 1) runs at
  the head of K3_i / HA_i instead of vLLM's op,
  from the previous launch's q and the layer's KV cache, block table and
  seq_lens (``kernels/attn_frag.py``).

Every rank must take the same path (the kernel waits on its peers), so the
decision uses only rank-invariant data: the static deployment checked at
attach, the batch shape and the attention metadata counts.
"""

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass
from types import ModuleType
from typing import Any

import torch
from torch import nn

from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.attention import get_attention_context
from vllm.models.muse_spark.common.layers import (
    MuseSpark1xNormPreservingResidualConnection,
    MuseSpark1xRMSNormWithInputScale,
    norm_then_gated_residual,
)

from ..muse_moe_runner import MuseSparkQuarkOCPMXFusedMoE
from .peer import PeerBuffer

logger = init_logger(__name__)

FIRST_LAYER = 1
LAST_LAYER = 60
ROWS_PER_REQUEST = 8
K3_ROWS = (8, 16)
K5_ROWS = (32, 64, 96, 128)
WORLD_SIZE = 8
NUM_CUS = 256
KV_BLOCK_SIZE = 16
HEALTH_POLL_INTERVAL = 64
# Attention types whose verify attention runs inside the launch, per build.
# Other builds and layer 1 run vLLM's attention op.
FUSED_ATTENTION = {8: ("swa", "full"), 16: ("swa", "full"), 32: ("swa", "full")}
SLIDING_WINDOW = (2047, 0)

# The model shape the K3 build is specialized for.
KERNEL_CONFIG: dict[str, float] = {
    "hidden_size": 8192,
    "moe_hidden_size": 4096,
    "moe_intermediate_size": 4096,
    "num_local_experts": 256,
    "num_experts_per_tok": 8,
    "head_dim": 64,
    "num_attention_heads": 128,
    "num_key_value_heads": 16,
    "num_hidden_layers": 62,
    "routed_scaling_factor": 1.0,
    "rms_norm_eps": 1e-5,
    "post_norm_eps": 1e-8,
    "residual_gate_temperature": 0.3,
}


@dataclass(frozen=True)
class MonoDims:
    """Per-rank tensor sizes of a TP = EP rank."""

    world_size: int
    hidden_size: int
    pooled_size: int
    intermediate_size: int
    num_experts: int
    num_heads: int
    num_kv_heads: int
    head_dim: int

    @classmethod
    def from_config(cls, config: Any, world_size: int) -> "MonoDims":
        return cls(
            world_size=world_size,
            hidden_size=config.hidden_size,
            pooled_size=config.moe_hidden_size,
            intermediate_size=config.moe_intermediate_size,
            num_experts=config.num_local_experts,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
        )

    @property
    def local_experts(self) -> int:
        return self.num_experts // self.world_size

    @property
    def local_kv_heads(self) -> int:
        return self.num_kv_heads // self.world_size

    @property
    def q_rows(self) -> int:
        return self.num_heads // self.world_size * self.head_dim

    @property
    def kv_rows(self) -> int:
        return self.local_kv_heads * self.head_dim

    @property
    def qkvg_rows(self) -> int:
        return 2 * self.q_rows + 2 * self.kv_rows

    def expert_shapes(self) -> dict[str, tuple[int, ...]]:
        """MXFP4 expert tensors of the local experts, as uint8."""
        e, h, i = self.local_experts, self.pooled_size, self.intermediate_size
        return {
            "w13_weight": (e, 2 * i, h // 2),
            "w13_weight_scale": (e, 2 * i, h // 32),
            "w2_weight": (e, h, i // 2),
            "w2_weight_scale": (e, h, i // 32),
        }


def bucket_of(rows: int) -> int | None:
    """The kernel build that serves ``rows`` (padded up); None = stock."""
    if rows <= 0 or rows % ROWS_PER_REQUEST:
        return None
    if rows in K3_ROWS:
        return rows
    return next((s for s in K5_ROWS if rows <= s), None)


@dataclass
class LayerWeights:
    """Views of vLLM's parameters (and small norm copies) for layer i's launches."""

    w_o: torch.Tensor
    alpha_a: torch.Tensor
    beta_a: torch.Tensor
    w_pffn: torch.Tensor
    w_r: torch.Tensor
    bias: torch.Tensor
    w_pre: torch.Tensor
    w_pren: torch.Tensor
    w13: torch.Tensor
    w13s: torch.Tensor
    w2: torch.Tensor
    w2s: torch.Tensor
    w_post: torch.Tensor
    w_pp: torch.Tensor
    w_pff: torch.Tensor
    alpha: torch.Tensor
    beta: torch.Tensor
    w_in: torch.Tensor
    w_qkvg: torch.Tensor
    next_rope: bool

    def _ptrs(self, *names: str) -> tuple[int, ...]:
        return tuple(getattr(self, n).data_ptr() for n in names)

    def ha_pointers(self) -> tuple[int, ...]:
        """Attention boundary, router and pre-expert projection (HA, K3)."""
        return self._ptrs(
            "w_o", "alpha_a", "beta_a", "w_pffn", "w_r", "bias", "w_pre", "w_pren"
        )

    def hm_pointers(self) -> tuple[int, ...]:
        """The local experts (HM, K3)."""
        return self._ptrs("w13", "w13s", "w2", "w2s", "w_post")

    def hb_pointers(self) -> tuple[int, ...]:
        """Post-expert projection, FFN boundary and the next QKV (HB, K3)."""
        return self._ptrs("w_pp", "w_pff", "alpha", "beta", "w_in", "w_qkvg")

    def k3_pointers(self) -> tuple[int, ...]:
        return self.ha_pointers() + self.hm_pointers() + self.hb_pointers()

    def expert_replica_pointers(self) -> tuple[int, ...]:
        """The EPLB replica arguments; without replicas, the own experts."""
        return self._ptrs("w13", "w13s", "w2", "w2s", "w13")


def effective_norm_weight(norm: nn.Module) -> torch.Tensor:
    """The bf16 multiplier the kernel applies: ``weight (+ gain_center)``."""
    weight = norm.weight.detach().float()
    if norm.zero_centered_gamma:
        weight = weight + norm.gain_center
    return weight.to(torch.bfloat16).contiguous()


def _routed_experts(layer: nn.Module) -> nn.Module | None:
    experts = getattr(layer.feed_forward, "experts", None)
    return getattr(experts, "routed_experts", None)


def _rows(t: torch.Tensor, rank: int, n: int) -> torch.Tensor:
    return t.detach()[rank * n : (rank + 1) * n]


def collect_layer_weights(
    layers: Sequence[nn.Module], i: int, rank: int, dims: MonoDims
) -> LayerWeights:
    """Layer i's kernel weights from decoder layers ``i`` and ``i + 1``; no copy."""
    layer, nxt = layers[i], layers[i + 1]
    moe = layer.feed_forward
    experts = _routed_experts(layer)
    assert experts is not None
    w = dims.world_size
    return LayerWeights(
        w_o=layer.self_attn.o_proj.weight.detach(),
        alpha_a=layer.add_attn.alpha,
        beta_a=layer.add_attn.beta,
        w_pffn=effective_norm_weight(layer.pre_ffn_norm),
        w_r=_rows(moe.router.weight, rank, dims.num_experts // w),
        bias=moe.expert_bias,
        w_pre=_rows(moe.pooler.weight, rank, dims.pooled_size // w),
        w_pren=effective_norm_weight(moe.pre_expert_norm),
        w13=experts.w13_weight.detach().view(torch.uint8),
        w13s=experts.w13_weight_scale.detach().view(torch.uint8),
        w2=experts.w2_weight.detach().view(torch.uint8),
        w2s=experts.w2_weight_scale.detach().view(torch.uint8),
        w_post=effective_norm_weight(moe.post_expert_norm),
        w_pp=_rows(moe.unpooler.weight, rank, dims.hidden_size // w),
        w_pff=effective_norm_weight(layer.post_ffn_norm),
        alpha=layer.add_ffn.alpha,
        beta=layer.add_ffn.beta,
        w_in=effective_norm_weight(nxt.self_attn.pre_qkv_proj_norm),
        w_qkvg=nxt.self_attn.qkv_proj.weight.detach(),
        next_rope=nxt.self_attn.rotary_emb is not None,
    )


def config_refusals(config: Any) -> list[str]:
    reasons = []
    for name, want in KERNEL_CONFIG.items():
        got = getattr(config, name, None)
        if not isinstance(got, (int, float)) or abs(got - want) > 1e-12 * max(
            1.0, abs(want)
        ):
            reasons.append(f"config.{name}={got} (kernel: {want})")
    return reasons


def deployment_refusals(
    vllm_config: Any, tp_rank: int, tp_size: int, moe_parallel: Any
) -> list[str]:
    """Parallel layout and server features the monokernel does not support."""
    reasons = []
    parallel = vllm_config.parallel_config
    if tp_size != WORLD_SIZE:
        reasons.append(f"tensor_parallel_size={tp_size} (kernel: {WORLD_SIZE})")
    for name in (
        "pipeline_parallel_size",
        "data_parallel_size",
        "decode_context_parallel_size",
        "prefill_context_parallel_size",
    ):
        if getattr(parallel, name, 1) != 1:
            reasons.append(f"{name}={getattr(parallel, name)}")
    if not parallel.enable_expert_parallel:
        reasons.append("expert parallelism off (needs --enable-expert-parallel)")
    if parallel.enable_eplb:
        reasons.append("EPLB on")
    if parallel.expert_placement_strategy != "linear":
        reasons.append(f"expert placement {parallel.expert_placement_strategy}")
    if moe_parallel is None or not moe_parallel.use_ep:
        reasons.append("MoE layers not expert parallel")
    elif moe_parallel.ep_size != tp_size or moe_parallel.ep_rank != tp_rank:
        reasons.append(
            f"EP rank {moe_parallel.ep_rank}/{moe_parallel.ep_size} "
            f"!= TP rank {tp_rank}/{tp_size}"
        )
    spec = vllm_config.speculative_config
    if spec is None or spec.num_speculative_tokens != ROWS_PER_REQUEST - 1:
        reasons.append(
            f"needs speculative decoding with {ROWS_PER_REQUEST - 1} draft tokens"
        )
    if vllm_config.lora_config is not None:
        reasons.append("LoRA on")
    if vllm_config.kv_transfer_config is not None:
        reasons.append("KV connector on")
    cache = vllm_config.cache_config
    if getattr(cache, "kv_offloading_size", None):
        reasons.append("KV offloading on")
    if cache.cache_dtype not in ("auto", "bfloat16"):
        reasons.append(f"kv_cache_dtype={cache.cache_dtype} (kernel: bf16)")
    if vllm_config.model_config.dtype != torch.bfloat16:
        reasons.append(f"model dtype {vllm_config.model_config.dtype}")
    return reasons


def _shape(t: torch.Tensor | None) -> tuple[int, ...] | None:
    return None if t is None else tuple(t.shape)


def layer_refusals(model: nn.Module, dims: MonoDims) -> list[str]:
    """Module and tensor guards for layers FIRST_LAYER .. LAST_LAYER + 1."""
    reasons: list[str] = []
    if getattr(model, "residual_dtype", None) != torch.float32:
        reasons.append("residual dtype not fp32")
    if not getattr(model, "fuse_across_layers", False):
        reasons.append("layer fusion off")
    layers = model.layers
    if len(layers) != LAST_LAYER + 2:
        return reasons + [f"{len(layers)} layers"]
    bf16 = torch.bfloat16
    for i in range(FIRST_LAYER, LAST_LAYER + 2):
        layer, attn = layers[i], layers[i].self_attn
        bad = reasons.append
        if not (
            attn.use_qk_norm
            and attn.use_o_norm
            and attn.use_o_gate
            and attn.fuse_o_norm_gate
            and attn.fused_qk_norm_rope_kv_insert
        ) or getattr(attn, "use_attn_temperature_tuning", False):
            bad(f"layer {i}: attention options (needs the fused KV insert)")
        backend = getattr(getattr(attn.attn, "attn_backend", None), "__name__", None)
        if backend != "AiterFlashAttentionBackend":
            bad(f"layer {i}: attention backend {backend}")
        w = attn.qkv_proj.weight
        if _shape(w) != (dims.qkvg_rows, dims.hidden_size) or w.dtype != bf16:
            bad(f"layer {i}: qkv_proj {_shape(w)} {w.dtype}")
        elif not w.is_contiguous():
            bad(f"layer {i}: qkv_proj not contiguous")
        rope = attn.rotary_emb
        if rope is not None and (
            rope.cos_sin_cache.dtype != torch.float32
            or rope.cos_sin_cache.shape[-1] != dims.head_dim
            or rope.rotary_dim != dims.head_dim
            or rope.is_neox_style
        ):
            bad(f"layer {i}: RoPE cache {_shape(rope.cos_sin_cache)}")
        if not layer.self_attn.pre_qkv_proj_norm.has_weight:
            bad(f"layer {i}: weightless pre_qkv_proj_norm")
        if i > LAST_LAYER:
            continue
        w = attn.o_proj.weight
        if _shape(w) != (dims.hidden_size, dims.q_rows) or w.dtype != bf16:
            bad(f"layer {i}: o_proj {_shape(w)} {w.dtype}")
        elif not w.is_contiguous():
            bad(f"layer {i}: o_proj not contiguous")
        for name in ("add_attn", "add_ffn"):
            conn = getattr(layer, name)
            if not isinstance(
                conn, MuseSpark1xNormPreservingResidualConnection
            ) or not getattr(conn, "_coefficients_ready", False):
                bad(f"layer {i}: {name} coefficients not materialized")
        if layer.post_attn_norm.has_weight:
            bad(f"layer {i}: post_attn_norm has a weight")
        if not isinstance(layer.post_ffn_norm, MuseSpark1xRMSNormWithInputScale):
            bad(f"layer {i}: post_ffn_norm type")
        if not getattr(layer, "is_moe_layer", False):
            bad(f"layer {i}: dense FFN")
            continue
        moe = layer.feed_forward
        r = moe.router.weight
        if _shape(r) != (dims.num_experts, dims.hidden_size) or r.dtype != (
            torch.float32
        ):
            bad(f"layer {i}: router {_shape(r)} {r.dtype}")
        if moe.router_output_multiplier is not None and not getattr(
            moe, "_router_multiplier_folded", False
        ):
            bad(f"layer {i}: router multiplier not folded")
        for name, shape in (
            ("pooler", (dims.pooled_size, dims.hidden_size)),
            ("unpooler", (dims.hidden_size, dims.pooled_size)),
        ):
            w = getattr(moe, name).weight
            if _shape(w) != shape or w.dtype != bf16 or not w.is_contiguous():
                bad(f"layer {i}: {name} {_shape(w)} {w.dtype}")
        experts = _routed_experts(layer)
        if experts is None or not isinstance(
            getattr(experts, "quant_method", None), MuseSparkQuarkOCPMXFusedMoE
        ):
            bad(f"layer {i}: experts not MXFP4 W4A4")
            continue
        for name, want in dims.expert_shapes().items():
            t = getattr(experts, name)
            if _shape(t) != want or t.element_size() != 1 or not t.is_contiguous():
                bad(f"layer {i}: {name} {_shape(t)} {t.dtype}")
        if not (
            getattr(experts.w13_weight, "is_shuffled", False)
            and getattr(experts.w2_weight, "is_shuffled", False)
        ):
            bad(f"layer {i}: expert weights not shuffled")
        for e_n in (2 * dims.intermediate_size, dims.pooled_size):
            if dims.local_experts * e_n % 256:
                bad(f"layer {i}: padded scale shuffle ({dims.local_experts} x {e_n})")
    return reasons


def kv_cache_refusal(kv_cache: torch.Tensor, kv_heads: int) -> str | None:
    """K3 writes bf16 LBHNC pages: ``[block][kv head][16 tokens][K | V]``."""
    if kv_cache.numel() == 0:
        return "kv:unbound"
    if kv_cache.dtype != torch.bfloat16:
        return f"kv:{kv_cache.dtype}"
    want_shape = (kv_heads, KV_BLOCK_SIZE, 2 * 64)
    if tuple(kv_cache.shape[1:]) != want_shape:
        return f"kv:shape{tuple(kv_cache.shape)}"
    if tuple(kv_cache.stride()[1:]) != (KV_BLOCK_SIZE * 128, 128, 1):
        return f"kv:stride{tuple(kv_cache.stride())}"
    return None


def attention_type(layer: nn.Module) -> str | None:
    """``"swa"`` / ``"full"`` when the fused attention can run layer's
    attention (vLLM's ``sliding_window`` (2047, 0) or none); else None."""
    window = getattr(
        getattr(layer.self_attn.attn, "impl", None), "sliding_window", None
    )
    if window == SLIDING_WINDOW:
        return "swa"
    if window in (None, (-1, -1)):
        return "full"
    return None


def fused_attention_layers(
    layers: Sequence[nn.Module], max_model_len: int, max_keys: int
) -> dict[int, str]:
    """Layers whose attention the launches can fuse: ``{i: type}``.

    Empty when a request may hold more keys than a build stages block ids for.
    """
    if max_model_len > max_keys:
        return {}
    fused = {}
    for i in range(FIRST_LAYER + 1, LAST_LAYER + 1):
        kind = attention_type(layers[i])
        if kind is not None:
            fused[i] = kind
    return fused


def batch_refusal(
    num_tokens: int, forward_context: Any, layer_names: Sequence[str]
) -> str | None:
    """Why this step runs stock, from rank-invariant data only; None = mono.

    Reads the attention metadata of the kernels' layers (``layer_names``) only:
    the dict also holds the draft model's layers.
    """
    if bucket_of(num_tokens) is None:
        return f"rows:{num_tokens}"
    mode = forward_context.cudagraph_runtime_mode
    if mode not in (CUDAGraphMode.NONE, CUDAGraphMode.FULL):
        return f"cudagraph:{mode.name}"
    metadata = forward_context.attn_metadata
    if not isinstance(metadata, dict) or not metadata:
        return "attn-metadata"
    seen: set[int] = set()
    for name in layer_names:
        md = metadata.get(name)
        if md is None:
            return "attn-metadata"
        if id(md) in seen:
            continue
        seen.add(id(md))
        decode = getattr(md, "decode_metadata", None)
        if (
            getattr(md, "num_prefills", None) != 0
            or getattr(md, "num_extends", None) != 0
            or decode is None
        ):
            return "not-decode"
        if decode.max_query_len != ROWS_PER_REQUEST:
            return f"query-len:{decode.max_query_len}"
        if md.num_decodes * ROWS_PER_REQUEST != num_tokens:
            return f"width:{num_tokens}/{md.num_decodes}"
    return None


@dataclass(frozen=True)
class MonoKernels:
    """The FlyDSL kernel modules (``kernels.k3``, ``k5``, ``k5m``, ``attn_frag``)."""

    k3: ModuleType
    k5: ModuleType
    k5m: ModuleType
    attn_frag: ModuleType


@dataclass(frozen=True)
class _LayerArgs:
    ha: tuple[int, ...]
    hm: tuple[int, ...]
    hb: tuple[int, ...]
    replicas: tuple[int, ...]
    next_rope: bool


class _FusedAttention:
    """The fused attention of one build: its output, scratch and descriptor
    tables (``kernels/attn_frag.py``; row i = layer i).

    A table holds pointers and counts only (q, KV cache, block table,
    seq_lens, output, request count, scale). Tables are keyed by their
    contents, so one is written once and never changes after a CUDA graph
    captured a launch that reads it.
    """

    def __init__(
        self,
        attn_frag: ModuleType,
        layers: dict[int, str],
        scales: dict[int, float],
        rows: int,
        dims: MonoDims,
        device: torch.device,
    ) -> None:
        self.af = attn_frag
        self.layers = layers
        self.scales = scales
        self.out = torch.zeros(rows, dims.q_rows, dtype=torch.bfloat16, device=device)
        self.part = torch.zeros(
            attn_frag.part_bytes() // 4, dtype=torch.int32, device=device
        )
        self.words = attn_frag.TABLE_ROWS * attn_frag.DESC_WORDS
        self.device = device
        self.tables: dict[tuple, torch.Tensor] = {}
        # Pinned sources of the table copies captured in CUDA graphs: a pinned
        # allocation is not allowed while a stream captures.
        pin = device.type == "cuda"
        self._pinned = [
            torch.zeros(self.words, dtype=torch.int64, pin_memory=pin) for _ in range(8)
        ]
        self._keep: list[torch.Tensor] = []

    def table(
        self, layers: Sequence[nn.Module], q_of: Any, num_reqs: int
    ) -> torch.Tensor:
        """The descriptor table of this step; ``q_of(i)`` = layer i's q."""
        rows = {}
        for i in self.layers:
            md, _, kv_cache, _ = get_attention_context(
                layers[i].self_attn.attn.layer_name
            )
            rows[i] = self.af.vllm_desc(
                q_of(i),
                kv_cache,
                md.block_table,
                md.seq_lens,
                self.out,
                self.part,
                num_reqs,
                self.scales[i],
            )
        key = tuple((i, *w) for i, w in rows.items())
        table = self.tables.get(key)
        if table is not None:
            return table
        host = torch.zeros(self.words, dtype=torch.int64)
        n = self.af.DESC_WORDS
        for i, w in rows.items():
            host[i * n : (i + 1) * n] = torch.tensor(w, dtype=torch.int64)
        table = torch.zeros(self.words, dtype=torch.int64, device=self.device)
        capturing = self.device.type == "cuda" and (
            torch.cuda.is_current_stream_capturing()
        )
        if capturing:
            if not self._pinned:
                raise RuntimeError(
                    "Muse Spark monokernel: more fused-attention tables than "
                    "pinned buffers"
                )
            pinned = self._pinned.pop()
            pinned.copy_(host)
            self._keep.append(pinned)
            table.copy_(pinned, non_blocking=True)
        else:
            table.copy_(host)
        self.tables[key] = table
        return table

    def row(self, table: torch.Tensor, i: int) -> int:
        return table.data_ptr() + i * self.af.DESC_WORDS * table.element_size()


class _Bucket:
    """Launchers and persistent buffers of one kernel build (row count)."""

    def __init__(
        self,
        kernel: ModuleType,
        rows: int,
        peer_bytes: int,
        dims: MonoDims,
        device: torch.device,
        group: Any,
    ) -> None:
        self.rows = rows
        self.dims = dims
        self.blocks = kernel.BLOCKS
        self.peers = PeerBuffer(peer_bytes + 256, group)
        self.epoch = torch.zeros(4, dtype=torch.int32, device=device)
        self.health = torch.zeros(kernel.HEALTH_WORDS, dtype=torch.int32, device=device)
        self.timeline = torch.zeros(
            kernel.BLOCKS * kernel.TL_POINTS, dtype=torch.int64, device=device
        )
        self.no_rope = torch.zeros(64, dtype=torch.float32, device=device)
        self.fa: _FusedAttention | None = None

    def _common(self) -> tuple[int, ...]:
        return (
            self.epoch.data_ptr(),
            self.health.data_ptr(),
            self.timeline.data_ptr(),
            *self.peers.pointers,
        )

    def run(
        self,
        mono: "MuseSparkMono",
        layers: Sequence[nn.Module],
        positions: torch.Tensor,
        o: torch.Tensor,
        g: torch.Tensor,
        stream: torch.Tensor,
        aux_layers: Sequence[int],
        aux_hidden_states: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Layers FIRST_LAYER..LAST_LAYER; returns layer 61's (stream, q, g)."""
        raise NotImplementedError


class _K3Bucket(_Bucket):
    """8 / 16 rows: one K3 launch a layer."""

    def __init__(
        self,
        k3: ModuleType,
        rows: int,
        dims: MonoDims,
        device: torch.device,
        group: Any,
        fa: _FusedAttention | None = None,
        fused_kinds: Sequence[tuple[bool, str]] = (),
    ) -> None:
        w = dims.world_size
        super().__init__(k3, rows, k3.peer_layout(rows, w)[1], dims, device, group)
        self.launchers = {
            next_rope: k3.build(k3.build_vllm(S=rows, W=w, next_rope=next_rope))
            for next_rope in (True, False)
        }
        self.fa = fa
        self.fused_launchers = {
            (next_rope, kind): k3.build(
                k3.build_vllm(S=rows, W=w, next_rope=next_rope, attn=kind)
            )
            for next_rope, kind in fused_kinds
        }
        self.scratch = torch.zeros(
            k3.scratch_layout(rows, w)[1] // 4 + 64, dtype=torch.int32, device=device
        )
        bf16 = torch.bfloat16
        self.outs = [
            {
                "stream": torch.zeros(
                    rows, dims.hidden_size, dtype=torch.float32, device=device
                ),
                "xn": torch.zeros(rows, dims.hidden_size, dtype=bf16, device=device),
                "part": torch.zeros(rows, dims.pooled_size, dtype=bf16, device=device),
                "q": torch.zeros(rows, dims.q_rows, dtype=bf16, device=device),
                "g": torch.zeros(rows, dims.q_rows, dtype=bf16, device=device),
            }
            for _ in range(2)
        ]

    def run(self, mono, layers, positions, o, g, stream, aux_layers, aux_hidden_states):
        q = o
        fa = self.fa
        table = None
        if fa is not None:
            table = fa.table(
                layers,
                lambda i: self.outs[(i - 1) % 2]["q"],
                self.rows // ROWS_PER_REQUEST,
            )
        for i in range(FIRST_LAYER, LAST_LAYER + 1):
            kind = fa.layers.get(i) if fa is not None else None
            if fa is not None and kind is not None:
                attn_o = fa.row(table, i)
            else:
                if i > FIRST_LAYER:
                    o = layers[i].self_attn.attn(q, None, None)
                attn_o = o.data_ptr()
            kv_cache, slots, cos_sin = mono.next_kv(layers[i + 1], self.no_rope)
            a = mono.layer_args[i]
            out = self.outs[i % 2]
            launch = (
                self.launchers[a.next_rope]
                if kind is None
                else self.fused_launchers[(a.next_rope, kind)]
            )
            launch(
                attn_o,
                g.data_ptr(),
                stream.data_ptr(),
                *a.ha,
                *a.hm,
                *a.hb,
                out["stream"].data_ptr(),
                out["xn"].data_ptr(),
                out["part"].data_ptr(),
                out["q"].data_ptr(),
                out["g"].data_ptr(),
                kv_cache.data_ptr(),
                # The kernel takes the block stride (elements) in the V slot.
                kv_cache.stride(0),
                slots.data_ptr(),
                cos_sin.data_ptr(),
                positions.data_ptr(),
                self.scratch.data_ptr(),
                *self._common(),
                i,
                mono.rank,
                g.stride(0),
                *a.replicas,
                stream=torch.cuda.current_stream(),
            )
            stream, q, g = out["stream"], out["q"], out["g"]
            if i + 1 in aux_layers:
                aux_hidden_states.append(stream.clone())
        return stream, q, g


class _K5Bucket(_Bucket):
    """24-128 rows: HA (attention boundary, router, pre-expert projection),
    HM (the local experts) and HB (FFN boundary, next QKV) a layer; between
    the launches the residual stream lives token-sharded unless the build
    replicates it (``key.bnd``)."""

    def __init__(
        self,
        k5: ModuleType,
        k5m: ModuleType,
        rows: int,
        dims: MonoDims,
        device: torch.device,
        group: Any,
        fa: _FusedAttention | None = None,
        fused_kinds: Sequence[str] = (),
    ) -> None:
        w = dims.world_size
        key = k5.build_vllm(S=rows, W=w, next_rope=True)
        super().__init__(
            k5, rows, k5.peer_layout(rows, w, key.bnd)[1], dims, device, group
        )
        self.bnd = key.bnd
        self.launch_a, launch_b = k5.build(key)
        self.launch_b = {
            True: launch_b,
            False: k5.build(dataclasses.replace(key, next_rope=False))[1],
        }
        self.launch_m = k5m.build(k5m.build_vllm(S=rows, W=w))
        self.fa = fa
        self.fused_launch_a = {
            kind: k5.build(k5.build_vllm(S=rows, W=w, next_rope=True, attn=kind))[0]
            for kind in fused_kinds
        }
        i32, f32, bf16 = torch.int32, torch.float32, torch.bfloat16
        self.scratch = torch.zeros(
            k5.scratch_layout(rows, w, key.bnd)[1] // 4 + 64, dtype=i32, device=device
        )
        self.scratch_m = torch.zeros(
            k5m.scratch_layout(rows)[1] // 4 + 64, dtype=i32, device=device
        )
        self.timeline_m = torch.zeros(k5.BLOCKS * 8, dtype=torch.int64, device=device)
        topk = int(KERNEL_CONFIG["num_experts_per_tok"])
        self.h = torch.zeros(rows, dims.pooled_size, dtype=bf16, device=device)
        self.ids = torch.zeros(rows, topk, dtype=i32, device=device)
        self.scores = torch.zeros(rows, topk, dtype=f32, device=device)
        self.part = torch.zeros(rows, dims.pooled_size, dtype=bf16, device=device)
        sharded_rows = rows // w if key.bnd < 2 else rows
        self.ts = [
            torch.zeros(sharded_rows, dims.hidden_size, dtype=f32, device=device)
            for _ in range(2)
        ]
        self.sfull = torch.zeros(rows, dims.hidden_size, dtype=f32, device=device)
        self.q = [
            torch.zeros(rows, dims.q_rows, dtype=bf16, device=device) for _ in range(2)
        ]
        self.g = [
            torch.zeros(rows, dims.q_rows, dtype=bf16, device=device) for _ in range(2)
        ]

    def run(self, mono, layers, positions, o, g, stream, aux_layers, aux_hidden_states):
        nrows = stream.shape[0]
        hidden = self.dims.hidden_size
        cur = torch.cuda.current_stream()
        s_in, s_full = stream, 1
        q = o
        fa = self.fa
        table = None
        if fa is not None:
            table = fa.table(
                layers, lambda i: self.q[(i - 1) % 2], nrows // ROWS_PER_REQUEST
            )
        for i in range(FIRST_LAYER, LAST_LAYER + 1):
            kind = fa.layers.get(i) if fa is not None else None
            if fa is not None and kind is not None:
                attn_o = fa.row(table, i)
            else:
                if i > FIRST_LAYER:
                    o = layers[i].self_attn.attn(q, None, None)
                attn_o = o.data_ptr()
            kv_cache, slots, cos_sin = mono.next_kv(layers[i + 1], self.no_rope)
            a = mono.layer_args[i]
            launch_a = self.launch_a if kind is None else self.fused_launch_a[kind]
            launch_a(
                attn_o,
                g.data_ptr(),
                s_in.data_ptr(),
                *a.ha,
                self.h.data_ptr(),
                self.ids.data_ptr(),
                self.scores.data_ptr(),
                self.scratch.data_ptr(),
                *self._common(),
                i,
                mono.rank,
                g.stride(0),
                s_full,
                nrows,
                stream=cur,
            )
            self.launch_m(
                self.h.data_ptr(),
                self.ids.data_ptr(),
                self.scores.data_ptr(),
                *a.hm,
                self.part.data_ptr(),
                self.scratch_m.data_ptr(),
                self.epoch.data_ptr(),
                self.health.data_ptr(),
                self.timeline_m.data_ptr(),
                *self.peers.pointers,
                i,
                mono.rank,
                nrows,
                *a.replicas,
                stream=cur,
            )
            gather = int(i == LAST_LAYER or i + 1 in aux_layers or self.bnd == 1)
            ts, q_out, g_out = self.ts[i % 2], self.q[i % 2], self.g[i % 2]
            self.launch_b[a.next_rope](
                self.part.data_ptr(),
                *a.hb,
                ts.data_ptr(),
                self.sfull.data_ptr(),
                q_out.data_ptr(),
                g_out.data_ptr(),
                kv_cache.data_ptr(),
                kv_cache.stride(0),
                slots.data_ptr(),
                cos_sin.data_ptr(),
                positions.data_ptr(),
                self.scratch.data_ptr(),
                *self._common(),
                i,
                mono.rank,
                hidden,
                0,
                gather,
                nrows,
                stream=cur,
            )
            if self.bnd == 1:
                s_in, s_full = self.sfull, 1
            else:
                s_in, s_full = ts, int(self.bnd >= 2)
            q, g = q_out[:nrows], g_out[:nrows]
            if i + 1 in aux_layers:
                aux_hidden_states.append(self.sfull[:nrows].clone())
        return self.sfull[:nrows], q, g


def built_buckets(max_num_seqs: int) -> list[int]:
    """The builds a server with ``max_num_seqs`` requests can use."""
    max_rows = min(max_num_seqs, K5_ROWS[-1] // ROWS_PER_REQUEST) * ROWS_PER_REQUEST
    return sorted(
        {
            b
            for rows in range(ROWS_PER_REQUEST, max_rows + 1, ROWS_PER_REQUEST)
            if (b := bucket_of(rows)) is not None
        }
    )


class MuseSparkMono:
    """Runs decoder layers 1-60 of eligible verify steps through the kernels."""

    def __init__(
        self,
        model: nn.Module,
        kernels: MonoKernels,
        rank: int,
        group: Any,
        device: torch.device,
        dims: MonoDims,
        buckets: Sequence[int],
        fused_attention: dict[int, str] | None = None,
    ) -> None:
        self.rank = rank
        self.dims = dims
        self.device = device
        self.refresh_weights(model)
        fused_attention = fused_attention or {}
        self.buckets: dict[int, _Bucket] = {}
        for b in buckets:
            kinds = FUSED_ATTENTION.get(b, ())
            layers = {i: t for i, t in fused_attention.items() if t in kinds}
            fa = None
            if layers:
                scales = {
                    i: float(model.layers[i].self_attn.attn.impl.scale) for i in layers
                }
                fa = _FusedAttention(kernels.attn_frag, layers, scales, b, dims, device)
            if b in K3_ROWS:
                k3_kinds = sorted(
                    {(self.layer_args[i].next_rope, t) for i, t in layers.items()}
                )
                self.buckets[b] = _K3Bucket(
                    kernels.k3, b, dims, device, group, fa, k3_kinds
                )
            else:
                self.buckets[b] = _K5Bucket(
                    kernels.k5,
                    kernels.k5m,
                    b,
                    dims,
                    device,
                    group,
                    fa,
                    sorted(set(layers.values())),
                )
        self._validated: set[int] = set()
        self._fallbacks: set[str] = set()
        self._engaged: set[int] = set()
        self._steps = 0
        self._health_host = {
            b: torch.zeros_like(buf.health, device="cpu").pin_memory()
            for b, buf in self.buckets.items()
        }
        self._health_event: torch.cuda.Event | None = None

    def refresh_weights(self, model: nn.Module) -> None:
        """Re-take the views, e.g. after a weight reload replaced parameters."""
        self.weights = {
            i: collect_layer_weights(model.layers, i, self.rank, self.dims)
            for i in range(FIRST_LAYER, LAST_LAYER + 1)
        }
        self.layer_args = {
            i: _LayerArgs(
                ha=w.ha_pointers(),
                hm=w.hm_pointers(),
                hb=w.hb_pointers(),
                replicas=w.expert_replica_pointers(),
                next_rope=w.next_rope,
            )
            for i, w in self.weights.items()
        }

    def _layer_names(self, model: nn.Module) -> list[str]:
        return [
            model.layers[i].self_attn.attn.layer_name
            for i in range(FIRST_LAYER, LAST_LAYER + 2)
        ]

    def next_kv(
        self, layer: nn.Module, no_rope: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The KV cache, slots and RoPE table the kernels write ``layer``'s KV with."""
        kv_cache, slots = self._kv_of(layer)
        assert slots is not None
        rope = layer.self_attn.rotary_emb
        return kv_cache, slots, rope.cos_sin_cache if rope is not None else no_rope

    def _kv_of(self, layer: nn.Module) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, _, kv_cache, slot_mapping = get_attention_context(
            layer.self_attn.attn.layer_name
        )
        return kv_cache, slot_mapping

    def _step_refusal(
        self, model: nn.Module, positions: torch.Tensor, x: Any
    ) -> str | None:
        if not isinstance(x, torch.Tensor):
            return "input"
        rows = x.shape[0]
        reason = batch_refusal(rows, get_forward_context(), self._layer_names(model))
        if reason is not None:
            return reason
        if bucket_of(rows) not in self.buckets:
            return f"rows:{rows}"
        if positions.dtype != torch.int64 or positions.shape != (rows,):
            return "positions"
        for i in range(FIRST_LAYER + 1, LAST_LAYER + 2):
            kv_cache, slots = self._kv_of(model.layers[i])
            reason = kv_cache_refusal(kv_cache, self.dims.local_kv_heads)
            if reason is not None:
                return reason
            if (
                slots is None
                or slots.dtype != torch.int64
                or slots.shape[0] < rows
                or slots.stride(-1) != 1
            ):
                return "slot-mapping"
        return None

    def try_forward(
        self,
        model: nn.Module,
        positions: torch.Tensor,
        x: Any,
        aux_hidden_states: list[torch.Tensor],
    ) -> Any:
        """Returns the residual stream after the last layer, or None for stock."""
        reason = self._step_refusal(model, positions, x)
        if reason is not None:
            if reason not in self._fallbacks:
                self._fallbacks.add(reason)
                logger.debug("Muse Spark monokernel: stock step (%s)", reason)
            return None
        rows = x.shape[0]
        if rows not in self._engaged:
            self._engaged.add(rows)
            bucket = bucket_of(rows)
            logger.info(
                "Muse Spark monokernel engaged for %d rows (%s, layers %d-%d)%s",
                rows,
                f"K3 S={bucket}" if bucket in K3_ROWS else f"HA+HM+HB S={bucket}",
                FIRST_LAYER,
                LAST_LAYER,
                " in CUDA graph capture"
                if torch.cuda.is_current_stream_capturing()
                else "",
            )
        return self._forward(model, positions, x, aux_hidden_states)

    def _validate(
        self, attn_o: torch.Tensor, attn_g: torch.Tensor, stream: torch.Tensor
    ) -> None:
        """Check the stock layer-1 tensors the kernels read, once a row count."""
        rows = stream.shape[0]
        if rows in self._validated:
            return
        bf16, d = torch.bfloat16, self.dims
        checks = (
            ("attn_o", attn_o, bf16, (rows, d.q_rows)),
            ("stream", stream, torch.float32, (rows, d.hidden_size)),
        )
        for name, t, dtype, shape in checks:
            if t.dtype != dtype or tuple(t.shape) != shape or not t.is_contiguous():
                raise RuntimeError(
                    f"Muse Spark monokernel: {name} is {t.dtype} "
                    f"{tuple(t.shape)} {t.stride()}, expected {dtype} {shape}"
                )
        if (
            attn_g.dtype != bf16
            or tuple(attn_g.shape) != (rows, d.q_rows)
            or attn_g.stride(1) != 1
        ):
            raise RuntimeError(
                f"Muse Spark monokernel: o-gate is {attn_g.dtype} "
                f"{tuple(attn_g.shape)} {attn_g.stride()}"
            )
        self._validated.add(rows)

    def _forward(
        self,
        model: nn.Module,
        positions: torch.Tensor,
        x: torch.Tensor,
        aux_hidden_states: list[torch.Tensor],
    ) -> Any:
        layers = model.layers
        aux_layers = model.aux_hidden_state_layers
        rows = bucket_of(x.shape[0])
        assert rows is not None
        bucket = self.buckets[rows]
        bucket.epoch.add_(1)

        y = layers[0](positions, x)
        if 1 in aux_layers:
            aux_hidden_states.append(norm_then_gated_residual(y))
        from ..model import norm_residual_then_norm  # model.py imports this package

        attn = layers[FIRST_LAYER].self_attn
        attn_input, stream = norm_residual_then_norm(
            y, attn.pre_qkv_proj_norm, out_dtype=layers[FIRST_LAYER].default_dtype
        )
        qkv, _ = attn.qkv_proj(attn_input)
        q, _, _, g = attn._qk_norm_rope_kv_insert(qkv, positions)
        o = attn.attn(q, None, None)
        self._validate(o, g, stream)
        stream, q, g = bucket.run(
            self, layers, positions, o, g, stream, aux_layers, aux_hidden_states
        )

        last = layers[LAST_LAYER + 1]
        o = last.self_attn.attn(q, None, None)
        y = last.after_attention(stream, last.self_attn.back(o, g))
        if LAST_LAYER + 2 in aux_layers:
            aux_hidden_states.append(norm_then_gated_residual(y))
        return y

    def poll_health(self) -> None:
        """Raise if a launch timed out waiting for a peer rank.

        Reads the health words with an asynchronous copy every
        HEALTH_POLL_INTERVAL calls and checks the previous copy, so the step
        never waits for the device.
        """
        if torch.cuda.is_current_stream_capturing():
            return
        self._steps += 1
        if self._health_event is not None and self._health_event.query():
            self._health_event = None
            for b, host in self._health_host.items():
                if int(host[0]) != 0:
                    records = host[8:].view(self.buckets[b].blocks, 8)
                    stuck = records[records[:, 0] != 0][:8].tolist()
                    raise RuntimeError(
                        f"Muse Spark monokernel: rank {self.rank}, S={b}: a launch "
                        f"timed out waiting for a peer rank {stuck}"
                    )
        if self._health_event is None and self._steps % HEALTH_POLL_INTERVAL == 0:
            for b, buf in self.buckets.items():
                self._health_host[b].copy_(buf.health, non_blocking=True)
            self._health_event = torch.cuda.Event()
            self._health_event.record()

    def close(self) -> None:
        for buf in self.buckets.values():
            buf.peers.close()

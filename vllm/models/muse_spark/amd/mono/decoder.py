# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MuseSpark1x decode monokernel: layers 1-60 of a verify step as K3 launches.

Per eligible step (one request x 8 verify tokens, TP8 with expert parallelism):

- layer 0: stock.
- layer 1: stock attention input and attention front (QKV, QK-norm, RoPE, KV
  insert), then vLLM's attention op.
- layers 1-60: K3_i takes layer i's attention output, runs o-norm/gate,
  o_proj, the attention boundary, the MoE half with its EP all-reduce and the
  FFN boundary, then layer i+1's QKV+O-gate GEMV, QK-norm, RoPE and KV insert.
  vLLM's attention op on K3_i's q is layer i+1's attention.
- layer 61: vLLM's attention op on K3_60's q, then the stock tail.

Every rank must take the same path (the kernel waits on its peers), so the
decision uses only rank-invariant data: the static deployment checked at
attach, the batch shape and the attention metadata counts.
"""

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
K3_ROWS = (8,)
WORLD_SIZE = 8
NUM_CUS = 256
KV_BLOCK_SIZE = 16
HEALTH_POLL_INTERVAL = 64

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


@dataclass
class K3LayerWeights:
    """Views of vLLM's parameters (and small norm copies) for K3_i."""

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

    def pointers(self) -> tuple[int, ...]:
        """K3's weight arguments, in launch order."""
        return tuple(
            t.data_ptr()
            for t in (
                self.w_o,
                self.alpha_a,
                self.beta_a,
                self.w_pffn,
                self.w_r,
                self.bias,
                self.w_pre,
                self.w_pren,
                self.w13,
                self.w13s,
                self.w2,
                self.w2s,
                self.w_post,
                self.w_pp,
                self.w_pff,
                self.alpha,
                self.beta,
                self.w_in,
                self.w_qkvg,
            )
        )

    def expert_replica_pointers(self) -> tuple[int, ...]:
        """K3's EPLB replica arguments; without replicas, the own experts."""
        return (
            self.w13.data_ptr(),
            self.w13s.data_ptr(),
            self.w2.data_ptr(),
            self.w2s.data_ptr(),
            self.w13.data_ptr(),
        )


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
) -> K3LayerWeights:
    """K3_i's arguments from decoder layers ``i`` and ``i + 1``; no weight copy."""
    layer, nxt = layers[i], layers[i + 1]
    moe = layer.feed_forward
    experts = _routed_experts(layer)
    assert experts is not None
    w = dims.world_size
    return K3LayerWeights(
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


def batch_refusal(num_tokens: int, forward_context: Any) -> str | None:
    """Why this step runs stock, from rank-invariant data only; None = mono."""
    if num_tokens not in K3_ROWS:
        return f"rows:{num_tokens}"
    mode = forward_context.cudagraph_runtime_mode
    if mode not in (CUDAGraphMode.NONE, CUDAGraphMode.FULL):
        return f"cudagraph:{mode.name}"
    metadata = forward_context.attn_metadata
    if not isinstance(metadata, dict) or not metadata:
        return "attn-metadata"
    seen: set[int] = set()
    for md in metadata.values():
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


class _K3Buffers:
    """Launchers and persistent buffers of one row count."""

    def __init__(
        self,
        k3: ModuleType,
        rows: int,
        dims: MonoDims,
        rank: int,
        device: torch.device,
        group: Any,
    ) -> None:
        self.k3 = k3
        self.rows = rows
        self.launchers = {
            next_rope: k3.build(
                k3.build_vllm(S=rows, W=dims.world_size, next_rope=next_rope)
            )
            for next_rope in (True, False)
        }
        _, scratch_bytes = k3.scratch_layout(rows, dims.world_size)
        _, peer_bytes = k3.peer_layout(rows, dims.world_size)
        self.scratch = torch.zeros(
            scratch_bytes // 4 + 64, dtype=torch.int32, device=device
        )
        self.peers = PeerBuffer(peer_bytes + 256, group)
        self.epoch = torch.zeros(4, dtype=torch.int32, device=device)
        self.health = torch.zeros(k3.HEALTH_WORDS, dtype=torch.int32, device=device)
        self.timeline = torch.zeros(
            k3.BLOCKS * k3.TL_POINTS, dtype=torch.int64, device=device
        )
        self.no_rope = torch.zeros(64, dtype=torch.float32, device=device)
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
        self.rank = rank
        self.dims = dims


class MuseSparkMono:
    """Runs decoder layers 1-60 of eligible verify steps through K3."""

    def __init__(
        self,
        model: nn.Module,
        k3: ModuleType,
        rank: int,
        group: Any,
        device: torch.device,
        dims: MonoDims,
    ) -> None:
        self.rank = rank
        self.dims = dims
        self.device = device
        self.k3 = k3
        self.weights: dict[int, K3LayerWeights] = {}
        self.refresh_weights(model)
        self.buffers = {
            rows: _K3Buffers(k3, rows, dims, rank, device, group) for rows in K3_ROWS
        }
        self._validated: set[tuple[int, int]] = set()
        self._fallbacks: set[str] = set()
        self._engaged: set[int] = set()
        self._steps = 0
        self._health_host = {
            rows: torch.zeros_like(b.health, device="cpu").pin_memory()
            for rows, b in self.buffers.items()
        }
        self._health_event: torch.cuda.Event | None = None

    def refresh_weights(self, model: nn.Module) -> None:
        """Re-take the views, e.g. after a weight reload replaced parameters."""
        self.weights = {
            i: collect_layer_weights(model.layers, i, self.rank, self.dims)
            for i in range(FIRST_LAYER, LAST_LAYER + 1)
        }
        self._pointers = {
            i: (w.pointers(), w.expert_replica_pointers())
            for i, w in self.weights.items()
        }

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
        reason = batch_refusal(x.shape[0], get_forward_context())
        if reason is not None:
            return reason
        if positions.dtype != torch.int64 or positions.shape != (x.shape[0],):
            return "positions"
        for i in range(FIRST_LAYER + 1, LAST_LAYER + 2):
            kv_cache, slots = self._kv_of(model.layers[i])
            reason = kv_cache_refusal(kv_cache, self.dims.local_kv_heads)
            if reason is not None:
                return reason
            if (
                slots is None
                or slots.dtype != torch.int64
                or slots.shape[0] < x.shape[0]
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
            logger.info(
                "Muse Spark monokernel engaged for %d rows (layers %d-%d)%s",
                rows,
                FIRST_LAYER,
                LAST_LAYER,
                " in CUDA graph capture"
                if torch.cuda.is_current_stream_capturing()
                else "",
            )
        return self._forward(model, positions, x, aux_hidden_states)

    def _validate(
        self,
        key: tuple[int, int],
        attn_o: torch.Tensor,
        attn_g: torch.Tensor,
        stream: torch.Tensor,
    ) -> None:
        """Check K3's activation arguments at the first launches of a row count."""
        if key in self._validated:
            return
        rows = key[0]
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
        self._validated.add(key)

    def _launch(
        self,
        buf: _K3Buffers,
        i: int,
        attn_o: torch.Tensor,
        attn_g: torch.Tensor,
        stream: torch.Tensor,
        out: dict[str, torch.Tensor],
        kv_cache: torch.Tensor,
        slots: torch.Tensor,
        cos_sin: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        weights, replicas = self._pointers[i]
        launch = buf.launchers[self.weights[i].next_rope]
        launch(
            attn_o.data_ptr(),
            attn_g.data_ptr(),
            stream.data_ptr(),
            *weights,
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
            buf.scratch.data_ptr(),
            buf.epoch.data_ptr(),
            buf.health.data_ptr(),
            buf.timeline.data_ptr(),
            *buf.peers.pointers,
            i,
            self.rank,
            attn_g.stride(0),
            *replicas,
            stream=torch.cuda.current_stream(),
        )

    def _forward(
        self,
        model: nn.Module,
        positions: torch.Tensor,
        x: torch.Tensor,
        aux_hidden_states: list[torch.Tensor],
    ) -> Any:
        layers = model.layers
        aux_layers = model.aux_hidden_state_layers
        rows = x.shape[0]
        buf = self.buffers[rows]
        buf.epoch.add_(1)

        y = layers[0](positions, x)
        if 1 in aux_layers:
            aux_hidden_states.append(norm_then_gated_residual(y))
        first = layers[FIRST_LAYER]
        attn_input, stream = first.attention_input(y)
        q, k, v, g = first.self_attn.front(positions, attn_input)
        o = first.self_attn.core(q, k, v)
        assert g is not None
        for i in range(FIRST_LAYER, LAST_LAYER + 1):
            if i > FIRST_LAYER:
                o = layers[i].self_attn.core(q, None, None)
            if i <= FIRST_LAYER + 1:
                self._validate((rows, i), o, g, stream)
            nxt = layers[i + 1]
            kv_cache, slots = self._kv_of(nxt)
            assert slots is not None
            rope = nxt.self_attn.rotary_emb
            out = buf.outs[i % 2]
            self._launch(
                buf,
                i,
                o,
                g,
                stream,
                out,
                kv_cache,
                slots,
                rope.cos_sin_cache if rope is not None else buf.no_rope,
                positions,
            )
            stream, q, g = out["stream"], out["q"], out["g"]
            if i + 1 in aux_layers:
                aux_hidden_states.append(stream.clone())

        last = layers[LAST_LAYER + 1]
        o = last.self_attn.core(q, None, None)
        y = last.after_attention(stream, last.self_attn.back(o, g))
        if LAST_LAYER + 2 in aux_layers:
            aux_hidden_states.append(norm_then_gated_residual(y))
        return y

    def poll_health(self) -> None:
        """Raise if a K3 launch timed out waiting for a peer rank.

        Reads the health words with an asynchronous copy every
        HEALTH_POLL_INTERVAL calls and checks the previous copy, so the step
        never waits for the device.
        """
        if torch.cuda.is_current_stream_capturing():
            return
        self._steps += 1
        if self._health_event is not None and self._health_event.query():
            self._health_event = None
            for rows, host in self._health_host.items():
                if int(host[0]) != 0:
                    records = host[8:].view(self.k3.BLOCKS, 8)
                    stuck = records[records[:, 0] != 0][:8].tolist()
                    raise RuntimeError(
                        f"Muse Spark monokernel: rank {self.rank}, {rows} rows: "
                        f"K3 timed out waiting for a peer rank {stuck}"
                    )
        if self._health_event is None and self._steps % HEALTH_POLL_INTERVAL == 0:
            for rows, buf in self.buffers.items():
                self._health_host[rows].copy_(buf.health, non_blocking=True)
            self._health_event = torch.cuda.Event()
            self._health_event.record()

    def close(self) -> None:
        for buf in self.buffers.values():
            buf.peers.close()

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model side of the GLM-5.3-Flash MonoKernel spec.

A KDA layer is one launch for the whole decoder layer. A sparse-attention layer
runs vLLM's attention half unchanged (the attention mHC pre and the attention,
with o_proj's TP all-reduce off) and then one launch for the rest: the attention
TP reduce, the FFN mHC seam, the RMSNorm, the router and the MoE with its TP
reduce. Both take and return what ``Glm5NextDecoderLayer`` does."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import torch

from vllm.models.common.mono.spec import MonoContext, MonoStep
from vllm.models.glm5next.amd.mono import spec as flash_spec
from vllm.models.glm5next.amd.mono.weights import (
    experts_backend_reason,
    kda_weights,
    layer_weights,
)
from vllm.models.glm5next.amd.ops.mono.config import (
    GLOBAL_INTER,
    HC_MULT,
    HIDDEN,
    KDA_DIM,
    FlashShard,
)

if TYPE_CHECKING:
    from torch import nn

POLL_LIMIT = 20_000_000


def _kernel_layers(model: nn.Module) -> list[int]:
    return flash_spec.kernel_layers(model.config)


def refuse_model(model: nn.Module) -> list[str]:
    reasons = []
    for idx in _kernel_layers(model):
        layer = model.layers[idx]
        why = experts_backend_reason(layer)
        if why is not None:
            reasons.append(f"layer {idx}: {why}")
        if not getattr(layer, "mhc", False) or getattr(layer, "is_mtp_layer", False):
            reasons.append(f"layer {idx}: needs an mHC decoder layer")
        if getattr(layer, "is_sequence_parallel", False):
            reasons.append(f"layer {idx}: sequence-parallel layer")
        if not hasattr(layer.self_attn, "o_proj"):
            reasons.append(f"layer {idx}: needs the attention's o_proj")
        if flash_spec.is_kda(layer) and getattr(layer.self_attn, "num_spec", 0):
            reasons.append(f"layer {idx}: KDA with speculative decoding")
    return reasons


def pack(model: nn.Module, ctx: MonoContext) -> dict[int, Any]:
    inter = GLOBAL_INTER // ctx.tp_size
    packed, reasons = {}, []
    for idx in _kernel_layers(model):
        layer = model.layers[idx]
        try:
            kda = kda_weights(layer, ctx.tp_size) if flash_spec.is_kda(layer) else None
            packed[idx] = (layer_weights(layer, inter), kda)
        except ValueError as error:
            reasons.append(f"layer {idx}: {error}")
    if reasons:
        raise ValueError(
            "VLLM_ROCM_MONO_DECODE cannot run GLM-5.3-Flash: " + "; ".join(reasons)
        )
    kda = [_kda_name(model.layers[i]) for i in sorted(packed) if packed[i][1]]
    ctx.state["kda_names"] = kda[:1]
    return packed


def bind(ctx: MonoContext, caches: Mapping[str, torch.Tensor]) -> list[str]:
    """Each KDA layer's conv state ([slots, 3P, 3] dim first, else [slots, 3, 3P])
    and recurrent state [slots, H, 128, 128], kept for the launches."""
    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
    from vllm.models.glm5next.amd.ops.mono.ffn.op import FlashKdaState

    shard = FlashShard(ctx.tp_size)
    P, H, D = shard.kda_proj, shard.kda_heads, KDA_DIM
    reasons, states = [], {}
    conv_shape, conv_strides = (3 * P, 3), (3, 1)
    if not is_conv_state_dim_first():
        conv_shape, conv_strides = (3, 3 * P), (3 * P, 1)
    for key, conv in caches.items():
        idx, name = key.split(".")
        if name != "conv":
            continue
        rec = caches[f"{idx}.rec"]
        if tuple(conv.shape[1:]) != conv_shape or conv.stride()[1:] != conv_strides:
            reasons.append(f"layer {idx}: conv state {tuple(conv.shape)}")
        if tuple(rec.shape[1:]) != (H, D, D) or rec.stride()[1:] != (D * D, D, 1):
            reasons.append(f"layer {idx}: recurrent state {tuple(rec.shape)}")
        if conv.shape[0] != rec.shape[0]:
            reasons.append(f"layer {idx}: {conv.shape[0]} conv, {rec.shape[0]} slots")
        states[int(idx)] = FlashKdaState(conv, rec)
    if not reasons:
        ctx.state["kda_states"] = states
    return reasons


def eligible(ctx: MonoContext, step: MonoStep) -> str | None:
    cap = flash_spec.TP_MAX_WIDTH.get(ctx.tp_size)
    if cap is not None and step.width > cap:
        return f"more than {cap} rows at TP{ctx.tp_size}"
    for name in ctx.state.get("kda_names", ()):
        meta = step.attn_metadata[name]
        if getattr(meta, "num_spec_decodes", 0):
            return "speculative decode rows"
        indices = getattr(meta, "non_spec_state_indices_tensor", None)
        if indices is None or indices.shape[0] < step.tokens:
            return "no KDA state index a row"
    return None


def metadata_names(layers: Mapping[int, nn.Module]) -> list[str]:
    """The kernel layers' KDA attention entries of the forward context; every
    decode step carries their metadata."""
    from vllm.config import get_current_vllm_config

    ctx = get_current_vllm_config().compilation_config.static_forward_context
    attns = {id(layer.self_attn) for layer in layers.values()}
    return sorted(name for name, module in ctx.items() if id(module) in attns)


def _kda_name(layer: nn.Module) -> str:
    return layer.self_attn.prefix


class _Width:
    """Per-width state of one launch kind: the runtime, padding inputs and, for
    KDA, the step's state indices."""

    def __init__(self, runtime, width: int, device: torch.device) -> None:
        self.rt = runtime
        self.pad = (
            torch.zeros(width, HIDDEN, dtype=torch.bfloat16, device=device),
            torch.zeros(width, HC_MULT, HIDDEN, dtype=torch.bfloat16, device=device),
            torch.zeros(width, HC_MULT, 1, dtype=torch.float32, device=device),
            torch.zeros(width, HC_MULT, HC_MULT, dtype=torch.float32, device=device),
        )
        self.idx = torch.full((width,), -1, dtype=torch.int32, device=device)


def _runtime(ctx: MonoContext, width: int, launches: int, kda: bool):
    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
    from vllm.models.glm5next.amd.ops.mono.ffn.op import FlashFfnRuntime

    widths = ctx.state.setdefault("widths", {})
    if (width, kda) not in widths:
        first = next(iter(widths.values()), None)
        rt = FlashFfnRuntime(
            width,
            ctx.tp_rank,
            ctx.tp_size,
            ctx.cpu_group,
            GLOBAL_INTER // ctx.tp_size,
            launches,
            POLL_LIMIT,
            poll_early_out=False,
            flag_addr=None if first is None else first.rt.flag_addr,
            kda=kda,
            conv_dim_first=is_conv_state_dim_first(),
        )
        widths[(width, kda)] = _Width(rt, width, ctx.device)
        if first is None and rt.flags is not None:
            ctx.state["flags"] = rt.flags
    return widths[(width, kda)]


def _pad(w: _Width, step: MonoStep, ins):
    n, S = step.tokens, step.width
    if n < S:
        for dst, src in zip(w.pad, ins):
            dst[:n].copy_(src)
        return w.pad
    return tuple(t.contiguous() for t in ins)


class FlashFfnLayer:
    """One decoder layer's kernel at one width; ``ends`` the runtimes the step's
    last kernel layer advances."""

    def __init__(self, ctx: MonoContext, layer: nn.Module, op, w: _Width, ends):
        self.layer, self.op, self.w, self.ends = layer, op, w, ends

    def health(self) -> torch.Tensor:
        return self.w.rt.words

    def finish(self, step: MonoStep, out):
        for rt in self.ends:
            rt.advance_step()
        return tuple(t[: step.tokens] for t in out)

    def launch(self, step: MonoStep, hidden_states, residual, post, comb):
        layer = self.layer
        if post is None:
            raise RuntimeError("a Flash kernel layer needs the previous mHC state")
        residual, post, comb, x = layer.hc_fused_post_pre(
            hidden_states,
            residual,
            post,
            comb,
            layer.hc_attn_fn,
            layer.hc_attn_scale,
            layer.hc_attn_base,
            norm_weight=layer.input_layernorm.weight.data,
            norm_eps=layer.input_layernorm.variance_epsilon,
        )
        o_proj = layer.self_attn.o_proj
        o_proj.reduce_results = False
        try:
            x = layer.self_attn(hidden_states=x, positions=step.positions)
        finally:
            o_proj.reduce_results = True
        n = step.tokens
        ins = _pad(self.w, step, (x, residual, post.view(n, HC_MULT, 1), comb))
        return self.finish(step, self.op.launch(*ins))


class FlashKdaLayer(FlashFfnLayer):
    """One whole KDA decoder layer at one width; the step's first KDA layer
    copies the step's state indices. vLLM pads graph rows with the null block 0
    and this width's own padding rows are -1: the kernel skips both."""

    def __init__(self, ctx, layer, op, w, ends, idx: int, first: bool) -> None:
        super().__init__(ctx, layer, op, w, ends)
        self.ctx, self.idx, self.first = ctx, idx, first

    def launch(self, step: MonoStep, hidden_states, residual, post, comb):
        if post is None:
            raise RuntimeError("a Flash kernel layer needs the previous mHC state")
        n = step.tokens
        if self.first:
            meta = step.attn_metadata[_kda_name(self.layer)]
            self.w.idx[:n].copy_(meta.non_spec_state_indices_tensor[:n])
            if n < step.width:
                self.w.idx[n:].fill_(-1)
        state = self.ctx.state["kda_states"][self.idx]
        ins = _pad(
            self.w, step, (hidden_states, residual, post.view(n, HC_MULT, 1), comb)
        )
        return self.finish(step, self.op.launch(*ins, state, self.w.idx))


def build(ctx: MonoContext, packed: dict, layer: nn.Module, idx: int, width: int):
    from vllm.models.glm5next.amd.ops.mono.ffn.op import FlashFfnOp

    kda = packed[idx][1] is not None
    order = sorted(i for i in packed if (packed[i][1] is not None) == kda)
    w = _runtime(ctx, width, len(order), kda)
    op = FlashFfnOp(w.rt, packed[idx][0], order.index(idx), packed[idx][1])
    last = idx == max(packed)
    ends = []
    if last:
        for k in (False, True):
            if any((packed[i][1] is not None) == k for i in packed):
                n = sum((packed[i][1] is not None) == k for i in packed)
                ends.append(_runtime(ctx, width, n, k).rt)
    if not kda:
        return FlashFfnLayer(ctx, layer, op, w, ends)
    return FlashKdaLayer(ctx, layer, op, w, ends, idx, idx == order[0])

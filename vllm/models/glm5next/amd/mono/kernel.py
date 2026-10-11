# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model side of the GLM-5.3-Flash MonoKernel spec (FFN-layer mode).

A kernel layer runs vLLM's attention half unchanged (the attention mHC pre and the
attention, with o_proj's TP all-reduce off) and then one launch for the rest: the
attention TP reduce, the FFN mHC seam, the RMSNorm, the router and the MoE with
its TP reduce. It takes and returns what ``Glm5NextDecoderLayer`` does."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import torch

from vllm.models.common.mono.spec import MonoContext, MonoStep
from vllm.models.glm5next.amd.mono import spec as flash_spec
from vllm.models.glm5next.amd.mono.weights import experts_backend_reason, layer_weights
from vllm.models.glm5next.amd.ops.mono.config import GLOBAL_INTER, HC_MULT, HIDDEN

if TYPE_CHECKING:
    from torch import nn

POLL_LIMIT = 20_000_000
_DEBUG = os.environ.get("VLLM_MONO_FLASH_DEBUG") == "1"
_DEBUG_SEEN: dict[int, int] = {}


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
    return reasons


def pack(model: nn.Module, ctx: MonoContext) -> dict[int, Any]:
    inter = GLOBAL_INTER // ctx.tp_size
    packed, reasons = {}, []
    for idx in _kernel_layers(model):
        try:
            packed[idx] = layer_weights(model.layers[idx], inter)
        except ValueError as error:
            reasons.append(f"layer {idx}: {error}")
    if reasons:
        raise ValueError(
            "VLLM_ROCM_MONO_DECODE cannot run GLM-5.3-Flash: " + "; ".join(reasons)
        )
    return packed


def bind(ctx: MonoContext, caches: Mapping[str, torch.Tensor]) -> list[str]:
    return []


def eligible(ctx: MonoContext, step: MonoStep) -> str | None:
    return None


def metadata_names(layers: Mapping[int, nn.Module]) -> list[str]:
    """The kernel layers' KDA attention entries of the forward context; every
    decode step carries their metadata."""
    from vllm.config import get_current_vllm_config

    ctx = get_current_vllm_config().compilation_config.static_forward_context
    attns = {id(layer.self_attn) for layer in layers.values()}
    return sorted(name for name, module in ctx.items() if id(module) in attns)


class _Width:
    """Per-width state: the runtime, padding inputs, two output sets."""

    def __init__(self, runtime, width: int, device: torch.device) -> None:
        self.rt = runtime
        self.pad = (
            torch.zeros(width, HIDDEN, dtype=torch.bfloat16, device=device),
            torch.zeros(width, HC_MULT, HIDDEN, dtype=torch.bfloat16, device=device),
            torch.zeros(width, HC_MULT, 1, dtype=torch.float32, device=device),
            torch.zeros(width, HC_MULT, HC_MULT, dtype=torch.float32, device=device),
        )


def _runtime(ctx: MonoContext, width: int, launches: int):
    from vllm.models.glm5next.amd.ops.mono.ffn.op import FlashFfnRuntime

    widths = ctx.state.setdefault("widths", {})
    if width not in widths:
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
        )
        widths[width] = _Width(rt, width, ctx.device)
        if first is None and rt.flags is not None:
            ctx.state["flags"] = rt.flags
    return widths[width]


class FlashFfnLayer:
    """One decoder layer's kernel at one width."""

    def __init__(self, ctx: MonoContext, layer: nn.Module, op, w: _Width, last: bool):
        self.layer, self.op, self.w, self.last = layer, op, w, last

    def health(self) -> torch.Tensor:
        return self.w.rt.words

    def launch(self, step: MonoStep, hidden_states, residual, post, comb):
        layer, rt = self.layer, self.w.rt
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
        n, S = step.tokens, step.width
        ins = (x, residual, post.view(n, HC_MULT, 1), comb)
        if n < S:
            for dst, src in zip(self.w.pad, ins):
                dst[:n].copy_(src)
            ins = self.w.pad
        else:
            ins = tuple(t.contiguous() for t in ins)
        out = self.op.launch(*ins)
        if _DEBUG and not torch.cuda.is_current_stream_capturing():
            self._debug(ins, out, n)
        if self.last:
            rt.advance_step()
        return tuple(t[:n] for t in out)

    def _debug(self, ins, out, n):
        from vllm.distributed import tensor_model_parallel_all_reduce
        from vllm.logger import init_logger

        layer = self.layer
        x_red = tensor_model_parallel_all_reduce(ins[0][:n].clone())
        r2, p2, c2, xin = layer.hc_fused_post_pre(
            x_red,
            ins[1][:n],
            ins[2][:n],
            ins[3][:n],
            layer.hc_ffn_fn,
            layer.hc_ffn_scale,
            layer.hc_ffn_base,
            norm_weight=layer.post_attention_layernorm.weight.data,
            norm_eps=layer.post_attention_layernorm.variance_epsilon,
        )
        moe = layer.mlp(xin)
        sc = self.w.rt.scratch
        a = sc[self.w.rt.sc["a"] :].view(torch.int32)[: n * HIDDEN : 2]
        k_in = a.contiguous().view(torch.bfloat16).view(n, HIDDEN)

        def rel(k, v):
            k, v = k.float().reshape(-1), v.float().reshape(-1)
            return ((k - v).norm() / v.norm().clamp_min(1e-30)).item()

        from vllm.models.glm5next.amd.ops.mono.config import MOE_SLOTS, TOP_K

        logits = xin.float() @ layer.mlp.gate.weight.float().T
        scores = torch.sigmoid(logits)
        keys = scores + layer.mlp.gate.e_score_correction_bias
        want = torch.topk(keys, TOP_K).indices
        sel = sc[self.w.rt.sc["sel"] :].view(torch.int32)[: 2 * n * MOE_SLOTS : 2]
        got = sel.view(n, MOE_SLOTS)[:, 1:].long()
        same = sum(
            len(set(a.tolist()) & set(b.tolist())) for a, b in zip(got, want)
        ) / (n * TOP_K)
        shared = sel.view(n, MOE_SLOTS)[:, 0].tolist()

        if self.op.rt.rank == 0 and _DEBUG_SEEN.setdefault(self.op.layer, 0) < 2:
            _DEBUG_SEEN[self.op.layer] += 1
            init_logger(__name__).warning(
                "flash mono debug layer %d n=%d: routing overlap %.3f shared %s "
                "R' %.2e post %.2e comb %.2e "
                "ffn_in %.2e moe %.2e | |x_attn| %.3e |moe| %.3e |kernel x| %.3e",
                self.op.layer,
                n,
                same,
                shared,
                rel(out[1][:n], r2),
                rel(out[2][:n], p2.view(n, HC_MULT, 1)),
                rel(out[3][:n], c2),
                rel(k_in, xin),
                rel(out[0][:n], moe),
                x_red.float().norm().item(),
                moe.float().norm().item(),
                out[0][:n].float().norm().item(),
            )


def build(ctx: MonoContext, packed: dict, layer: nn.Module, idx: int, width: int):
    from vllm.models.glm5next.amd.ops.mono.ffn.op import FlashFfnOp

    order = sorted(packed)
    w = _runtime(ctx, width, len(order))
    op = FlashFfnOp(w.rt, packed[idx], order.index(idx))
    return FlashFfnLayer(ctx, layer, op, w, idx == order[-1])

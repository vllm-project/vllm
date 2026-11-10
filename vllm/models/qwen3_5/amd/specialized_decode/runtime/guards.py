# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The refusals that need the loaded model (``model_reasons``: the module
variants and every weight's dtype and layout as the kernels read it in place)
and the GDN states vLLM binds (``cache_reasons``), each listing every reason;
and ``FailStop``, which stops the engine once an in-kernel wait gave up."""

from typing import NamedTuple

import torch

from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.models.qwen3_5.amd.specialized_decode.runtime import NAME
from vllm.models.qwen3_5.amd.specialized_decode.wide import layout as L

FP4 = (torch.uint8, torch.float4_e2m1fn_x2)


class LinearFormat(NamedTuple):
    """How the kernels read a linear: ``fmt`` "bf16", "mxfp4" or "mxfp4_h32"
    (online MXFP4 of a 32-block Hadamard-rotated weight, the input rotated
    the same way before its quantization), ``shuffled`` the MXFP4 weight on
    AITER's ASM FP4 GEMM layout."""

    fmt: str
    shuffled: bool = False


def linear_format(linear) -> LinearFormat | None:
    """``linear``'s format, None for one the kernels do not read: unquantized
    (bf16), or online MXFP4 on AITER's FP4 GEMMs, optionally inside a
    32-block ``InputHadamardLinearMethod``."""
    from vllm.model_executor.kernels.linear.mxfp4.aiter import AiterMxfp4LinearKernel
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.model_executor.layers.quantization.input_hadamard import (
        InputHadamardLinearMethod,
    )
    from vllm.model_executor.layers.quantization.online.mxfp4 import (
        Mxfp4OnlineLinearMethod,
    )

    qm = linear.quant_method
    if isinstance(qm, UnquantizedLinearMethod):
        return LinearFormat("bf16")
    fmt = "mxfp4"
    if isinstance(qm, InputHadamardLinearMethod):
        if qm.size != 32 or not qm.rotate_weight:
            return None
        qm, fmt = qm.quant_method, "mxfp4_h32"
    if not (
        isinstance(qm, Mxfp4OnlineLinearMethod)
        and isinstance(qm.kernel, AiterMxfp4LinearKernel)
    ):
        return None
    return LinearFormat(fmt, bool(qm.kernel.use_asm_gemm))


def fp4_scale_shape(rows: int, k: int, shuffled: bool) -> tuple[int, int]:
    cols = k // 32
    return (rows, (cols + 7) // 8 * 8) if shuffled else (cols, rows)


def _linear_reasons(name: str, linear, rows: int, k: int) -> list[str]:
    f = linear_format(linear)
    if f is None:
        return [f"{name} neither bf16 nor online MXFP4 on AITER's FP4 GEMMs"]
    w = linear.weight
    if getattr(linear, "bias", None) is not None:
        return [f"{name} with a bias"]
    if f.fmt == "bf16":
        if tuple(w.shape) != (rows, k) or w.dtype != torch.bfloat16:
            return [f"{name} {tuple(w.shape)} {w.dtype}"]
        if not w.is_contiguous():
            return [f"{name} not contiguous"]
        return []
    ws = linear.weight_scale
    if tuple(w.shape) != (rows, k // 2) or w.dtype not in FP4 or not w.is_contiguous():
        return [f"{name} {tuple(w.shape)} {w.dtype}"]
    if (
        tuple(ws.shape) != fp4_scale_shape(rows, k, f.shuffled)
        or ws.element_size() != 1
        or not ws.is_contiguous()
    ):
        return [f"{name} scales {tuple(ws.shape)} {ws.dtype}"]
    return []


def gdn_states(attn) -> tuple[torch.Tensor, torch.Tensor]:
    """A GDN layer's (conv state [slots, 3, CONV_DIM], recurrent state [slots,
    NV, DV, DK]) as vLLM binds them (slot strides as they come)."""
    conv_state, rstate = attn.kv_cache
    return conv_state, rstate


def router_gate(mlp) -> torch.Tensor:
    """The router gate with the shared expert's gate as its last row, [E + 1,
    HIDDEN] bf16: the MoE runner's own fused copy (the stock fused shared
    expert path routes with it), made here if it does not exist yet."""
    runner = mlp.experts
    if getattr(runner, "_fse_fuse_gate", False):
        if runner._combined_gate_weight is None:
            runner.process_weights_after_loading()
        return runner._combined_gate_weight
    return torch.cat([mlp.gate.weight, mlp.shared_expert_gate.weight], 0).contiguous()


def routed_experts(mlp):
    return getattr(mlp.experts, "routed_experts", mlp.experts)


def experts_rip(mlp) -> int:
    return routed_experts(mlp).w13_weight.shape[1] // 2


def _compact(idx: list[int]) -> str:
    """[0, 1, 2, 5] -> '0-2, 5'."""
    runs: list[list[int]] = []
    for i in idx:
        if runs and i == runs[-1][1] + 1:
            runs[-1][1] = i
        else:
            runs.append([i, i])
    return ", ".join(f"{a}" if a == b else f"{a}-{b}" for a, b in runs)


def _moe_reasons(mlp) -> tuple[list[str], dict[str, torch.Tensor]]:
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import Mxfp4MoeBackend
    from vllm.model_executor.models.qwen3_next import Qwen3NextSparseMoeBlock

    if not isinstance(mlp, Qwen3NextSparseMoeBlock):
        return ["a dense MLP"], {}
    if not mlp.is_fused_shared_expert_enabled:
        return ["the shared expert not fused"], {}
    out = []
    ex = routed_experts(mlp)
    backend = getattr(ex.quant_method, "mxfp4_backend", None)
    rip = ex.w13_weight.shape[1] // 2
    if backend is not Mxfp4MoeBackend.AITER_MXFP4_MXFP4:
        out.append(f"routed experts on {backend}, not AITER_MXFP4_MXFP4")
    elif not getattr(ex.w13_weight, "is_shuffled", False):
        out.append("routed experts not shuffled")
    elif rip % 128 or rip < L.RI:
        out.append(f"expert intermediate padded to {rip}")
    elif tuple(ex.w13_weight.shape) != (L.E_ALL, 2 * rip, L.HIDDEN // 2):
        out.append(f"w13 {tuple(ex.w13_weight.shape)}")
    elif tuple(ex.w2_weight.shape) != (L.E_ALL, L.HIDDEN, rip // 2):
        out.append(f"w2 {tuple(ex.w2_weight.shape)}")
    elif tuple(ex.w13_weight_scale.shape[1:]) != (
        2 * rip,
        (L.HIDDEN // 32 + 7) // 8 * 8,
    ):
        out.append(f"w13 scales {tuple(ex.w13_weight_scale.shape)}")
    elif tuple(ex.w2_weight_scale.shape[1:]) != (L.HIDDEN, (rip // 32 + 7) // 8 * 8):
        out.append(f"w2 scales {tuple(ex.w2_weight_scale.shape)}")
    dense = {
        "gate": mlp.gate.weight,
        "shared_expert_gate": mlp.shared_expert_gate.weight,
    }
    return out, dense


def _gdn_reasons(a) -> tuple[list[str], dict[str, torch.Tensor]]:
    if not (
        hasattr(a, "in_proj_qkvz")
        and not a.disable_tp_for_ba_proj
        and a.qkvz_layout == "flat"
        and a.activation == "silu"
        and a.conv1d.bias is None
        and a.norm.activation in ("silu", "swish")
        and a.norm.group_size is None
        and a.norm.norm_before_gate
    ):
        return ["GDN layer variant"], {}
    out = _linear_reasons("in_proj_qkvz", a.in_proj_qkvz, L.QKVZ, L.HIDDEN)
    out += _linear_reasons("out_proj", a.out_proj, L.HIDDEN, L.VALUE_DIM)
    if linear_format(a.in_proj_ba) != LinearFormat("bf16"):
        out.append("in_proj_ba quantized")
    if a.A_log.dtype != torch.float32:
        out.append("A_log not fp32")
    dense = {
        "in_proj_ba": a.in_proj_ba.weight,
        "conv1d": a.conv1d.weight,
        "dt_bias": a.dt_bias,
        "norm": a.norm.weight,
    }
    return out, dense


def _attention_reasons(a) -> tuple[list[str], dict[str, torch.Tensor]]:
    from vllm.model_executor.models.qwen3_next import Qwen3NextAttention

    if not isinstance(a, Qwen3NextAttention):
        return ["attention variant"], {}
    if (a.num_heads, a.num_kv_heads, a.head_dim) != (L.HEADS, L.KV_HEADS, L.ATTN_DIM):
        return [f"attention heads {a.num_heads}/{a.num_kv_heads}x{a.head_dim}"], {}
    rope = a.rotary_emb
    if not (
        a.attn_output_gate
        and getattr(rope, "is_neox_style", False)
        and getattr(rope, "dtype", None) in (torch.float16, torch.bfloat16)
    ):
        return ["attention without the gated NeoX RoPE front"], {}
    out = []
    for name, linear, rows, k in (
        ("qkv_proj", a.qkv_proj, L.QKV, L.HIDDEN),
        ("o_proj", a.o_proj, L.HIDDEN, L.ATTN_CORE),
    ):
        if linear_format(linear) != LinearFormat("bf16"):
            out.append(f"{name} quantized (bf16 only)")
        else:
            out += _linear_reasons(name, linear, rows, k)
    return out, {}


def _layer_reasons(layer) -> list[str]:
    if getattr(layer, "layer_scale", False):
        return ["layer scale"]
    if layer.use_attn_reduce_scatter_for_moe:
        return ["sequence-parallel MoE"]
    out, dense = _moe_reasons(layer.mlp)
    dense |= {
        "input_layernorm": layer.input_layernorm.weight,
        "post_attention_layernorm": layer.post_attention_layernorm.weight,
    }
    if layer.layer_type == "linear_attention":
        why, more = _gdn_reasons(layer.linear_attn)
    else:
        why, more = _attention_reasons(layer.self_attn)
    out += why
    dense |= more
    out += [
        f"{name} {w.dtype}, not bf16" for name, w in dense.items() if w.dtype != torch.bfloat16
    ]
    out += [f"{name} not contiguous" for name, w in dense.items() if not w.is_contiguous()]
    return out


def model_reasons(model, tp: int) -> list[str]:
    """Every reason the kernels cannot serve the loaded ``Qwen3_5Model``."""
    reasons = []
    if tp != L.TP:
        reasons.append(f"TP {tp}")
    if model.start_layer != 0 or model.end_layer != len(model.layers):
        reasons.append("partial layers")
    if model.aux_hidden_state_layers:
        reasons.append("auxiliary hidden state outputs")
    if model.layers[0].layer_type != "linear_attention":
        reasons.append("a first layer that is not a GDN layer")
    for a, b in zip(model.layers, model.layers[1:]):
        if a.layer_type != "linear_attention" and b.layer_type != "linear_attention":
            reasons.append("two consecutive full-attention layers")
            break
    if model.embed_tokens.weight.dtype != torch.bfloat16:
        reasons.append("embedding not bf16")
    if model.norm.weight.dtype != torch.bfloat16:
        reasons.append("final norm not bf16")
    rips = {experts_rip(ly.mlp) for ly in model.layers if hasattr(ly.mlp, "experts")}
    if len(rips) > 1:
        reasons.append(f"experts padded differently by layer: {sorted(rips)}")
    by_reason: dict[str, list[int]] = {}
    for layer in model.layers:
        for why in _layer_reasons(layer):
            by_reason.setdefault(why, []).append(layer.layer_idx)
    reasons += [f"{why} (layers {_compact(idx)})" for why, idx in by_reason.items()]
    return reasons


def cache_reasons(model, tp: int) -> list[str]:
    """Every reason the kernels cannot update the GDN states vLLM bound."""
    by_reason: dict[str, list[int]] = {}
    if is_conv_state_dim_first():
        return ["conv states stored dim-first"]
    for layer in model.layers:
        if layer.layer_type != "linear_attention":
            continue
        conv_state, rstate = gdn_states(layer.linear_attn)
        why = []
        if (
            conv_state.dtype != torch.bfloat16
            or tuple(conv_state.shape[1:]) != (L.CONV_TAPS - 1, L.CONV_DIM)
            or tuple(conv_state.stride()[1:]) != (L.CONV_DIM, 1)
        ):
            why.append(
                f"conv state {conv_state.dtype} {tuple(conv_state.shape[1:])} "
                f"{tuple(conv_state.stride()[1:])}, needs bf16 dense "
                f"({L.CONV_TAPS - 1}, {L.CONV_DIM})"
            )
        if (
            rstate.dtype != torch.float32
            or tuple(rstate.shape[1:]) != (L.NV, L.GDN_DV, L.GDN_DK)
            or not rstate[0].is_contiguous()
        ):
            why.append(
                f"recurrent state {rstate.dtype} {tuple(rstate.shape[1:])}, needs "
                f"dense fp32 ({L.NV}, {L.GDN_DV}, {L.GDN_DK})"
            )
        for w in why:
            by_reason.setdefault(w, []).append(layer.layer_idx)
    return [f"{why} (layers {_compact(idx)})" for why, idx in by_reason.items()]


class FailStop:
    """Stops every TP rank at the same step once any of them gave up an in-kernel
    wait (its health word FAILED, ``wide.abi``). Each step queues a D2H copy of
    this rank's word and keeps the newest copy that has landed, without blocking;
    every CHECK_EVERY steps the ranks agree on those over the TP CPU group and all
    raise together. A rank that raised alone would leave the others in the next
    collective for ever. A failure is seen within about CHECK_EVERY steps."""

    CHECK_EVERY = 16

    def __init__(self, failed: torch.Tensor, rank: int, world: int, group):
        self.failed, self.rank, self.world, self.group = failed, rank, world, group
        self.host = torch.zeros(1, dtype=torch.int32, pin_memory=True)
        self.copied: torch.cuda.Event | None = None
        self.seen = False
        self.steps = 0

    def after_step(self) -> None:
        """Once per step on every TP rank (compute_logits, itself collective)."""
        if torch.cuda.is_current_stream_capturing():
            return
        if self.copied is None or self.copied.query():
            if self.copied is not None:
                self.seen |= bool(self.host[0])
            self.host.copy_(self.failed, non_blocking=True)
            self.copied = torch.cuda.Event()
            self.copied.record()
        self.steps += 1
        if self.steps % self.CHECK_EVERY:
            return
        votes = torch.zeros(self.world, dtype=torch.int32)
        votes[self.rank] = int(self.seen)
        torch.distributed.all_reduce(votes, group=self.group)
        if votes.any():
            ranks = [r for r, v in enumerate(votes.tolist()) if v]
            raise RuntimeError(
                f"{NAME}: TP ranks {ranks} gave up an in-kernel wait; stopping the "
                "engine (without --monokernel, vLLM's decode runs)"
            )

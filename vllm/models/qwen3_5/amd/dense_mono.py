# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm FlyDSL dense mono decode for Qwen3.8-27B (``VLLM_ROCM_MONO_DECODE=1``).

Ported from SGLang #43558. The same flag selects DeepSeek-V4.1 mono decode on
that model; on ``qwen3_5_text`` it selects this dense 27B path. With the flag
unset, nothing here is constructed.

A pure decode step of a few rows runs the rank-local GDN mixer (K1) and the
rank-local MXFP4 FFN (K2). Norms, full attention, and the tensor-parallel
all-reduces stay on the stock path. There is no fused all-reduce.

TP1 and TP2 each have a separately compiled mono graph. The uncompiled
outer forward calls it only for a pure decode of 1..cap rows. Prefill, mixed
batches, and wider decode call the original compiled model, which is not
wrapped. TP1's graph folds the norms into K1/K2. TP2's graph keeps the norms
and the full-attention block visible to mode 3, and all-reduces each rank's
K1 and K2 partial with the same collective ``RowParallelLinear`` uses, so
FlyDSL QuickReduce still sees that op. Prefill, mixed batches, spec decode,
pipeline parallel, and TP other than 1 or 2 stay on the stock path. A set
flag on a shape, quant, or parallel setup these kernels cannot run raises
and lists the reason.

The launches are custom ops, as in vLLM #60833's split between the compiled
model and the mono step. ``torch.compile`` (compilation mode 3) traces the
mono graph, including RMSNorm and the all-reduces, and keeps K1/K2 as one
opaque call. A ``@torch.compiler.disable`` wrapper is skipped during that
trace, which is why mode 3 used to fail closed.
"""

import os

import torch
from torch import nn

import vllm.envs as envs
from vllm.compilation.decorators import support_torch_compile
from vllm.distributed import get_pp_group, get_tp_group
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

_CHECK_SLOTS = 4
# K1 sums in another order than the stock kernels and can differ from them
# by a bf16 ulp, so it must be about as close to an fp32 reference as stock
# is. K2 stays bit exact.
_K1_COS_SLACK = 1.25
_K1_COS_FLOOR = 1e-6
_K1_ABS_SLACK = 2.0
# Default row caps by TP. Chosen on MI355X against the AITER ASM FP4 GEMM.
_DEFAULT_ROWS = {1: (4, 4), 2: (8, 8)}
_K1_ROWS = "VLLM_ROCM_QWEN38_DENSE_MONO_K1_MAX_ROWS"
_K2_ROWS = "VLLM_ROCM_QWEN38_DENSE_MONO_K2_MAX_ROWS"


def _flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.lower() in ("1", "true", "yes")


def _optional_int(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    return int(raw)


def _is_gdn(layer) -> bool:
    return hasattr(layer, "linear_attn")


def _o_proj(layer):
    return layer.linear_attn.out_proj if _is_gdn(layer) else layer.self_attn.o_proj


def row_caps(tp: int) -> tuple[int, int]:
    """K1's and K2's widest step at this TP: the defaults, unless the
    ``VLLM_ROCM_QWEN38_DENSE_MONO_K1_MAX_ROWS`` / ``_K2_MAX_ROWS`` overrides
    are set. Clamped to the kernels' builds (8 at TP1, 16 at TP2)."""
    from vllm.models.qwen3_5.amd.mono import dense_ffn as D

    top = D.max_tokens(tp)
    caps = []
    for name, default in zip((_K1_ROWS, _K2_ROWS), _DEFAULT_ROWS[tp]):
        cap = _optional_int(name)
        if cap is None:
            cap = default
        elif not 1 <= cap <= top:
            logger.warning(
                "%s=%d: not in 1..%d at TP%d, using %d",
                name,
                cap,
                top,
                tp,
                min(max(cap, 1), top),
            )
            cap = min(max(cap, 1), top)
        caps.append(cap)
    return caps[0], caps[1]


def _unsupported(model, vllm_config) -> str | None:
    """Why these kernels cannot serve this dense model, else None."""
    try:
        from vllm.models.qwen3_5.amd.mono import dense_ffn as D
        from vllm.models.qwen3_5.amd.mono.common.plan import BLOCKS
    except ImportError as exc:
        return f"FlyDSL is not available ({exc})"

    from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP

    c = model.config
    if get_tp_group().world_size not in (1, 2):
        return "needs TP1 or TP2"
    if not current_platform.is_rocm():
        return "needs a gfx950 GPU"
    from vllm.platforms.rocm import on_gfx950

    if not on_gfx950():
        return "needs gfx950"
    if get_pp_group().world_size != 1:
        return "pipeline parallel"
    if current_platform.num_compute_units() < BLOCKS:
        return f"fewer than {BLOCKS} compute units (a partitioned GPU)"
    connector = getattr(vllm_config.kv_transfer_config, "kv_connector", None)
    if connector:
        return f"KV connector {connector}"
    if vllm_config.lora_config is not None:
        return "LoRA"
    if (
        c.hidden_size != D.HIDDEN
        or c.intermediate_size != D.INTER
        or c.hidden_act != "silu"
    ):
        return "not the Qwen3.8-27B shape"
    if any(not isinstance(layer.mlp, Qwen2MoeMLP) for layer in model.layers):
        return "not a dense MLP"
    if any(getattr(layer, "layer_scale", False) for layer in model.layers):
        return "layer scale"
    if any(layer.use_attn_reduce_scatter_for_moe for layer in model.layers):
        return "sequence parallel"
    if any(_is_gdn(layer) and layer.linear_attn.num_spec for layer in model.layers):
        return "speculative tokens widen the GDN conv state"
    return None


def _mxfp4_unsupported(proj, n: int, k: int) -> str | None:
    """Why ``proj`` is not a plain Quark MXFP4 ``[n, k]`` linear, else None."""
    from vllm.model_executor.layers.quantization.quark.quark import QuarkLinearMethod
    from vllm.model_executor.layers.quantization.quark.schemes.quark_ocp_mx import (
        QuarkOCP_MX,
    )

    scheme = getattr(proj, "scheme", None)
    if not isinstance(getattr(proj, "quant_method", None), QuarkLinearMethod):
        return "not Quark MXFP4"
    if not isinstance(scheme, QuarkOCP_MX) or scheme.weight_dtype != "mxfp4":
        return "not Quark MXFP4"
    w, ws = proj.weight, getattr(proj, "weight_scale", None)
    for marker in ("is_shuffled", "aiter_bpreshuffled"):
        if getattr(w, marker, False) or getattr(ws, marker, False):
            return f"weights already preshuffled ({marker})"
    if w.dtype != torch.uint8 or tuple(w.shape) != (n, k // 2):
        return f"weight {tuple(w.shape)} {w.dtype}, want MXFP4 {n} x {k}"
    if ws is None or tuple(ws.shape) != (n, k // 32) or not ws.is_contiguous():
        return "weight scales not plain e8m0 [N, K / 32]"
    return None


def _weights_unsupported(layer, oproj: bool, tp: int) -> str | None:
    from vllm.models.qwen3_5.amd.mono import dense_ffn as D

    inter = D.FGeo(tp).inter
    projs = [
        ("gate_up_proj", layer.mlp.gate_up_proj, 2 * inter, D.HIDDEN),
        ("down_proj", layer.mlp.down_proj, D.HIDDEN, inter),
    ]
    if oproj:
        projs.append(("o_proj", _o_proj(layer), D.HIDDEN, D.CORE))
    for name, proj, n, k in projs:
        why = _mxfp4_unsupported(proj, n, k)
        if why is not None:
            return f"{name}: {why}"
    return None


def _k1_weights_unsupported(attn, tp: int) -> str | None:
    from vllm.models.qwen3_5.amd.mono import dense_gdn as K

    geo = K.Geo(tp)
    if attn.tp_size != tp:
        return f"GDN heads split {attn.tp_size} ways at TP{tp}"
    if attn.gqa_interleaved_layout:
        return "interleaved GDN projection layout"
    for name, proj, n in (
        ("in_proj_qkvz", attn.in_proj_qkvz, geo.qkvz),
        ("in_proj_ba", attn.in_proj_ba, geo.ba),
    ):
        why = _mxfp4_unsupported(proj, n, K.HIDDEN)
        if why is not None:
            return f"{name}: {why}"
    weight = attn.conv1d.weight
    if weight.dtype != torch.bfloat16 or weight.numel() != geo.conv * K.CONV_W:
        return "conv1d weight"
    if not weight.is_contiguous() or attn.conv1d.bias is not None:
        return "conv1d weight not contiguous or with bias"
    if attn.activation != "silu":
        return f"conv activation {attn.activation}"
    if attn.A_log.dtype != torch.float32 or attn.dt_bias.dtype != torch.bfloat16:
        return f"A_log {attn.A_log.dtype} / dt_bias {attn.dt_bias.dtype}"
    if attn.A_log.numel() != geo.nv or attn.dt_bias.numel() != geo.nv:
        return f"A_log / dt_bias of {attn.A_log.numel()} heads, want {geo.nv}"
    gated = (
        attn.norm.weight.dtype == torch.bfloat16
        and attn.norm.activation in ("silu", "swish")
        and attn.norm.group_size is None
        and attn.norm.norm_before_gate
    )
    if not gated:
        return "gated norm variant"
    if (attn.num_k_heads, attn.num_v_heads, attn.head_k_dim, attn.head_v_dim) != (
        K.NK,
        K.NV,
        K.HD,
        K.HD,
    ):
        return "head shape"
    return None


def _ref_linear(x, w, ws):
    """Triton ``gemm_afp4wfp4`` on the checkpoint layout. The kernels match
    this GEMM; the prefill path may use the ASM GEMM instead."""
    from aiter.ops.triton.gemm_afp4wfp4 import gemm_afp4wfp4
    from aiter.ops.triton.quant import dynamic_mxfp4_quant

    xq, xs = dynamic_mxfp4_quant(x)
    y = torch.empty(x.size(0), w.size(0), dtype=x.dtype, device=x.device)
    gemm_afp4wfp4(xq, w, xs, ws, x.dtype, y)
    return y


def _ref_silu_and_mul(x):
    """The stock ROCm SiLU-mul custom op.

    ``SiluAndMul()`` dispatches to its native PyTorch implementation when
    vLLM compile selects ``custom_ops: none``, but to this op in eager mode.
    Dense K2 follows this op's bf16 rounding in both modes.
    """
    out = torch.empty(*x.shape[:-1], x.shape[-1] // 2, dtype=x.dtype, device=x.device)
    torch.ops._C.silu_and_mul(out, x)
    return out


_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _mxfp4_f32(packed, scale):
    """[N, K/2] packed e2m1 (even element in the low nibble) and [N, K/32]
    e8m0 scales -> fp32 [N, K]."""
    lut = torch.tensor(
        _E2M1 + tuple(-v for v in _E2M1), dtype=torch.float32, device=packed.device
    )
    p = packed.view(torch.uint8)
    v = torch.stack([lut[(p & 0xF).long()], lut[(p >> 4).long()]], -1).flatten(-2)
    sc = torch.exp2(scale.view(torch.uint8)[: v.size(0)].float() - 127)
    return (v.view(v.size(0), -1, 32) * sc[..., None]).view(v.size(0), -1)


def _f32_linear(x, w, ws):
    """``_ref_linear`` with the same MXFP4 operands, summed in fp32 and not
    rounded."""
    from aiter.ops.triton.quant import dynamic_mxfp4_quant

    xq, xs = dynamic_mxfp4_quant(x.bfloat16())
    return _mxfp4_f32(xq, xs) @ _mxfp4_f32(w, ws).T


def _k1_f32(attn, geo, x, conv, ssm, idx):
    """The stock GDN decode step in fp32 torch; updates ``conv`` / ``ssm`` (fp32
    copies) in place and returns (core, out_proj partial)."""
    from vllm.model_executor.layers.layernorm import RMSNormGated

    s = x.size(0)
    nk = attn.num_k_heads // attn.tp_size
    kd, vd = attn.head_k_dim, attn.head_v_dim
    qkv_size = (attn.key_dim * 2 + attn.value_dim) // attn.tp_size
    qkvz = _f32_linear(x, attn.in_proj_qkvz.weight, attn.in_proj_qkvz.weight_scale)
    ba = _f32_linear(x, attn.in_proj_ba.weight, attn.in_proj_ba.weight_scale)
    qkv, z = qkvz[:, :qkv_size], qkvz[:, qkv_size:]
    b, a = ba.chunk(2, dim=-1)
    w = attn.conv1d.weight.view(attn.conv1d.weight.size(0), -1).float()
    core = torch.zeros(s, geo.nv * vd, dtype=torch.float32, device=x.device)
    for r in range(s):
        slot = int(idx[r])
        if slot < 0:
            continue
        win = torch.cat([conv[slot], qkv[r, :, None]], -1)
        conv[slot] = win[:, 1:]
        mixed = torch.nn.functional.silu((win[:, -w.size(1) :] * w).sum(-1))
        q = mixed[: nk * kd].view(nk, kd)
        k = mixed[nk * kd : 2 * nk * kd].view(nk, kd)
        v = mixed[2 * nk * kd :].view(geo.nv, vd)
        q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6) * kd**-0.5
        k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
        rep = geo.nv // nk
        q, k = q.repeat_interleave(rep, 0), k.repeat_interleave(rep, 0)
        g = -torch.exp(attn.A_log.float()) * torch.nn.functional.softplus(
            a[r] + attn.dt_bias.float()
        )
        h = ssm[slot] * torch.exp(g)[:, None, None]
        v = (v - (h * k[:, None, :]).sum(-1)) * torch.sigmoid(b[r])[:, None]
        h = h + v[:, :, None] * k[:, None, :]
        ssm[slot] = h
        o = (h * q[:, None, :]).sum(-1)
        core[r] = RMSNormGated.forward_static(
            o,
            z[r].view(geo.nv, vd),
            attn.norm.weight,
            attn.norm.eps,
            torch.float32,
            None,
            True,
            attn.norm.activation,
        ).flatten()
    out = _f32_linear(core, attn.out_proj.weight, attn.out_proj.weight_scale)
    return core, out


def _min_row_cosine(got, ref) -> float:
    """The smallest per-row cosine of two [rows, n] tensors; NaN if any is not
    finite."""
    a, b = got.float(), ref.float()
    if not (torch.isfinite(a).all() and torch.isfinite(b).all()):
        return float("nan")
    return torch.nn.functional.cosine_similarity(a, b, dim=-1).min().item()


def _f32_distance(got, f32) -> tuple[float, float]:
    """(1 - smallest row cosine, max abs error) of ``got`` against fp32."""
    a = got.flatten(1)
    return (
        1 - _min_row_cosine(a, f32.flatten(1)),
        (a.float() - f32.flatten(1)).abs().max().item(),
    )


def _k1_verdict(s, live, res, pairs) -> str | None:
    """``res`` is (K1, stock); ``pairs`` maps a name to (K1, stock, fp32)."""
    if res[0] is not None and not torch.equal(*res):
        return f"K1 res_out at width {s}"
    pad = pairs["core"][0][~live]
    if not torch.equal(pad, torch.zeros_like(pad)):
        return f"K1 core pad rows at width {s} not zero"
    for name, (got, ref, f32) in pairs.items():
        if got is None:
            continue
        rows = live if name in ("core", "out_proj partial") else slice(None)
        k_cos, k_abs = _f32_distance(got[rows], f32[rows])
        s_cos, s_abs = _f32_distance(ref[rows], f32[rows])
        logger.debug(
            "Qwen3.8 dense mono K1 %s at width %d vs fp32: 1-cos %.3g max abs "
            "%.3g; stock 1-cos %.3g max abs %.3g",
            name,
            s,
            k_cos,
            k_abs,
            s_cos,
            s_abs,
        )
        if not (
            k_cos <= _K1_COS_SLACK * s_cos + _K1_COS_FLOOR
            and k_abs <= _K1_ABS_SLACK * s_abs
        ):
            return (
                f"K1 {name} at width {s} vs fp32: 1-cos {k_cos:.3g} max abs "
                f"{k_abs:.3g}; stock 1-cos {s_cos:.3g} max abs {s_abs:.3g}"
            )
    return None


def _gemma(x, residual, weight, eps):
    from vllm.models.minimax_m3.amd.ops.gemma_rmsnorm import (
        gemma_fused_add_rmsnorm,
        gemma_rmsnorm,
    )

    if residual is None:
        return gemma_rmsnorm(x, weight, eps), x
    return gemma_fused_add_rmsnorm(x, residual, weight, eps)


# One DenseMono per worker. The custom ops cannot take the module as an
# argument, so they read it back from here.
_BOUND: "DenseMono | None" = None


def _gdn_op(hidden: torch.Tensor, index: int) -> torch.Tensor:
    """This rank's out_proj partial. The mono graph all-reduces it."""
    mono = _BOUND
    layer = mono.model.layers[index]
    _, out = mono._k1_from_normed(index, layer, hidden)
    if out is None:
        raise RuntimeError("VLLM_ROCM_MONO_DECODE TP2 K1 did not fold out_proj")
    return out


def _gdn_op_fake(hidden: torch.Tensor, index: int) -> torch.Tensor:
    return torch.empty_like(hidden)


def _ffn_op(hidden: torch.Tensor, index: int) -> torch.Tensor:
    """This rank's down-proj partial. The mono graph all-reduces it."""
    mono = _BOUND
    return mono._ffn_partial(index, mono.model.layers[index], hidden)


def _ffn_op_fake(hidden: torch.Tensor, index: int) -> torch.Tensor:
    return torch.empty_like(hidden)


def _layer_op(
    hidden: torch.Tensor,
    residual: torch.Tensor,
    positions: torch.Tensor,
    index: int,
    first: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One TP1 decoder layer.

    This opaque op is the true branch of a compiled ``torch.cond``.  Its
    whole-layer K1/K2 composition is the same path used by the original TP1
    whole-step forward, while the false branch remains the compiled vLLM
    layer for widths above the mono cap.
    """
    mono = _BOUND
    return mono._layer(index, hidden, None if first else residual, positions)


def _layer_op_fake(
    hidden: torch.Tensor,
    residual: torch.Tensor,
    positions: torch.Tensor,
    index: int,
    first: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(hidden), torch.empty_like(hidden)


def _tail_op(
    core: torch.Tensor,
    residual: torch.Tensor,
    index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    mono = _BOUND
    if index == 0:
        rows = core.shape[0]
        if rows not in mono._logged:
            mono._logged.add(rows)
            logger.info("Qwen3.8 dense mono layer graph at width %d", rows)
        mono.epoch.add_(1)
    return mono._tail(index, mono.model.layers[index], core, residual, True)


def _tail_op_fake(
    core: torch.Tensor,
    residual: torch.Tensor,
    index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(residual), torch.empty_like(residual)


def _register_ops() -> None:
    from vllm.utils.torch_utils import direct_register_custom_op

    direct_register_custom_op(
        op_name="qwen38_dense_mono_gdn",
        op_func=_gdn_op,
        fake_impl=_gdn_op_fake,
    )
    direct_register_custom_op(
        op_name="qwen38_dense_mono_ffn",
        op_func=_ffn_op,
        fake_impl=_ffn_op_fake,
    )
    direct_register_custom_op(
        op_name="qwen38_dense_mono_layer",
        op_func=_layer_op,
        fake_impl=_layer_op_fake,
    )
    direct_register_custom_op(
        op_name="qwen38_dense_mono_tail",
        op_func=_tail_op,
        fake_impl=_tail_op_fake,
    )


_register_ops()


def _gdn_forward(mono, index, layer, stock, hidden_states):
    if not isinstance(hidden_states, torch.Tensor):
        return stock(hidden_states)
    return torch.ops.vllm.qwen38_dense_mono_gdn(hidden_states, index)


def _mlp_forward(mono, index, layer, stock, x):
    if not isinstance(x, torch.Tensor):
        return stock(x)
    return torch.ops.vllm.qwen38_dense_mono_ffn(x, index)


def _reduce(proj, partial):
    """The all-reduce the replaced ``RowParallelLinear`` would have run."""
    from vllm.distributed import tensor_model_parallel_all_reduce

    if proj.reduce_results and proj.tp_size > 1:
        return tensor_model_parallel_all_reduce(partial)
    return partial


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "inputs_embeds": 0,
    }
)
class DenseMonoGraph(nn.Module):
    """A separately compiled, branch-free mono entry point.

    The uncompiled outer model chooses this module only for an eligible width.
    Wider and non-decode steps continue to call the original compiled model.
    TP2 keeps full attention, the norms, and both all-reduces in this graph.
    """

    def __init__(self, mono, *, vllm_config):
        super().__init__()
        # Do not register the serving model a second time as a child module.
        object.__setattr__(self, "_mono", mono)

    def forward(self, input_ids, positions, inputs_embeds=None):
        mono = self._mono
        if mono.whole_step:
            return self._forward_tp1(mono, input_ids, positions, inputs_embeds)
        return self._forward_tp2(mono, input_ids, positions, inputs_embeds)

    def _forward_tp1(self, mono, input_ids, positions, inputs_embeds):
        model = mono.model
        hidden = (
            inputs_embeds
            if inputs_embeds is not None
            else model.embed_input_ids(input_ids)
        )
        residual = hidden
        for index, layer in enumerate(model.layers):
            if _is_gdn(layer):
                hidden, residual = torch.ops.vllm.qwen38_dense_mono_layer(
                    hidden, residual, positions, index, index == 0
                )
                continue
            if index == 0:
                residual = hidden
                x = layer.input_layernorm(hidden)
            else:
                x, residual = layer.input_layernorm(hidden, residual)
            attn = layer.self_attn
            qkv, _ = attn.qkv_proj(x)
            q, k, v, gate = attn._project_qkv_gate(qkv, positions)
            core = attn.attn(q, k, v)
            if gate is not None:
                core = core * torch.sigmoid(gate)
            hidden, residual = torch.ops.vllm.qwen38_dense_mono_tail(
                core, residual, index
            )
        hidden, _ = model.norm(hidden, residual)
        return hidden

    def _forward_tp2(self, mono, input_ids, positions, inputs_embeds):
        model = mono.model
        hidden = (
            inputs_embeds
            if inputs_embeds is not None
            else model.embed_input_ids(input_ids)
        )
        residual = None
        for index, layer in enumerate(model.layers):
            if residual is None:
                residual = hidden
                x = layer.input_layernorm(hidden)
            else:
                x, residual = layer.input_layernorm(hidden, residual)
            if _is_gdn(layer):
                partial = torch.ops.vllm.qwen38_dense_mono_gdn(x, index)
                hidden = _reduce(layer.linear_attn.out_proj, partial)
            else:
                hidden = layer.self_attn(positions=positions, hidden_states=x)
            if layer.layer_scale:
                hidden = hidden * (layer.attn_layer_scale.to(hidden.dtype)[0] + 1)
            x, residual = layer.post_attention_layernorm(hidden, residual)
            partial = torch.ops.vllm.qwen38_dense_mono_ffn(x, index)
            hidden = _reduce(layer.mlp.down_proj, partial)
            if layer.layer_scale:
                hidden = hidden * (layer.ffn_layer_scale.to(hidden.dtype)[0] + 1)
        hidden, _ = model.norm(hidden, residual)
        return hidden


def _oproj_forward(mono, stock, x, *args, **kwargs):
    if mono._skip_o:
        return x, None
    return stock(x, *args, **kwargs)


class DenseMono:
    """Bound to one Qwen3.5 text model; built at model init when the flag is set."""

    def __init__(self, model, vllm_config):
        global _BOUND
        self.model = model
        why = _unsupported(model, vllm_config)
        self.tp = get_tp_group().world_size
        self.whole_step = self.tp == 1
        self.oproj = self.whole_step and _flag("VLLM_ROCM_QWEN38_DENSE_MONO_OPROJ", True)
        k1_flag = _flag("VLLM_ROCM_QWEN38_DENSE_MONO_K1", True)
        self.k1 = k1_flag and (self.oproj or not self.whole_step)
        self.k1_oproj = (
            self.k1 and not self.whole_step and _flag("VLLM_ROCM_QWEN38_DENSE_MONO_OPROJ", True)
        )
        self.k2 = not self.whole_step and _flag("VLLM_ROCM_QWEN38_DENSE_MONO_K2", True)
        self.k1_rows, self.k2_rows = (0, 0) if why else row_caps(self.tp)
        self._skip_o = False
        self._gdn_step = False
        self._ffn_step = False
        self._prepared = False
        self._logged: set = set()
        self.w13 = []
        self.w2 = []
        self.w_o = []
        self.w_in = []
        if why is None and not self.whole_step and not (self.k1 or self.k2):
            why = "TP2 with neither K1 nor K2"
        if why is not None:
            raise ValueError(
                "VLLM_ROCM_MONO_DECODE cannot run Qwen3.8-27B dense mono decode: "
                + why
            )
        logger.info(
            "Qwen3.8 dense mono row caps at TP%d: K1 %d, K2 %d",
            self.tp,
            self.k1_rows,
            self.k2_rows,
        )
        self.scratch = None
        self.k1_scratch = None
        self.epoch = None
        # vLLM's GDN null slot is 0. The kernel's pad slot is -1.
        self.idx_buf = None
        _BOUND = self
        # TP1 and TP2. The stock module is not wrapped; this graph is a
        # second compiled entry the outer forward calls for an eligible width.
        self.graph = (
            DenseMonoGraph(self, vllm_config=vllm_config) if self.tp in (1, 2) else None
        )

    def _k1_key(self, dense_gdn, s: int, *, first: bool, eps=1e-6, norm_eps=1e-6):
        if self.whole_step:
            return dense_gdn.K1Build(tokens=s, first=first, eps=eps, norm_eps=norm_eps)
        return dense_gdn.K1Build(
            tokens=s, norm_eps=norm_eps, tp=self.tp, normed=True, oproj=self.k1_oproj
        )

    def _k2_key(self, dense_ffn, s: int, eps=1e-6):
        if self.whole_step:
            return dense_ffn.FfnBuild(tokens=s, eps=eps, shuffled=True, oproj=self.oproj)
        return dense_ffn.FfnBuild(tokens=s, shuffled=True, tp=self.tp, normed=True)

    def prepare(self) -> None:
        """After weights load, before the first decode: shuffle a private copy
        and check the kernels against the Triton reference. Ranks agree, then
        a failure raises on every rank."""
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "VLLM_ROCM_MONO_DECODE was first reached inside a CUDA graph "
                "capture, before an eager decode step."
            )
        why = self._prepare()
        if self.tp > 1:
            why = self._agree(why)
        if why is not None:
            self.w13, self.w2, self.w_o, self.w_in = [], [], [], []
            raise ValueError(
                "VLLM_ROCM_MONO_DECODE cannot run Qwen3.8-27B dense mono decode: "
                + why
            )
        self._prepared = True
        shuffled = sum(w.numel() for w in self.w13 + self.w2 + self.w_o if w is not None)
        shuffled += sum(a.numel() + b.numel() for a, b in filter(None, self.w_in))
        logger.info(
            "Qwen3.8 dense mono on for decode steps of <= %d rows at TP%d, "
            "o_proj %s, FFN %s, GDN K1 %s (%.1f GiB of shuffled weights)",
            self._max_tokens(),
            self.tp,
            "folded" if self.oproj else "in K1" if self.k1_oproj else "stock",
            f"K2 (<= {self.k2_rows} rows)" if self.whole_step or self.k2 else "stock",
            f"on (<= {self.k1_rows} rows)" if self.k1 else "off",
            shuffled / 2**30,
        )

    def _allocate(self) -> None:
        from vllm.models.qwen3_5.amd.mono import dense_ffn as D

        dev = self.model.layers[0].mlp.down_proj.weight.device
        if dev.type != "cuda":
            raise RuntimeError(
                "VLLM_ROCM_MONO_DECODE needs the loaded weights on the GPU "
                f"(they are on {dev})"
            )
        if self.whole_step or self.k2:
            self.scratch = torch.zeros(
                max(
                    D.scratch_bytes(self._k2_key(D, s))
                    for s in range(1, self.k2_rows + 1)
                ),
                dtype=torch.uint8,
                device=dev,
            )
        self.epoch = torch.zeros(1, dtype=torch.int32, device=dev)
        self.idx_buf = torch.empty(D.max_tokens(self.tp), dtype=torch.int32, device=dev)
        if self.k1:
            from vllm.models.qwen3_5.amd.mono import dense_gdn

            self.k1_scratch = torch.zeros(
                max(
                    dense_gdn.scratch_bytes(self._k1_key(dense_gdn, s, first=flag))
                    for s in range(1, self.k1_rows + 1)
                    for flag in (False, True)
                ),
                dtype=torch.uint8,
                device=dev,
            )

    def _prepare(self) -> str | None:
        from vllm.models.qwen3_5.amd.mono import dense_ffn as D

        for layer in self.model.layers:
            if self.whole_step or self.k2:
                why = _weights_unsupported(layer, self.oproj, self.tp)
                if why is not None:
                    return f"layer {layer.layer_idx} {why}"
            if self.k1 and _is_gdn(layer):
                why = _k1_weights_unsupported(layer.linear_attn, self.tp)
                if why is None and self.k1_oproj:
                    why = _mxfp4_unsupported(
                        layer.linear_attn.out_proj, D.HIDDEN, D.CORE // self.tp
                    )
                    why = None if why is None else f"out_proj: {why}"
                if why is not None:
                    if not self.whole_step:
                        return f"K1 needed at TP2: layer {layer.layer_idx} {why}"
                    self.k1 = False
                    logger.warning(
                        "Qwen3.8 dense mono K1 off: layer %d %s", layer.layer_idx, why
                    )
        self._allocate()
        self._shuffle_weights()
        return self._self_check()

    def _agree(self, why: str | None) -> str | None:
        ok = torch.tensor([int(why is None)], dtype=torch.int32)
        torch.distributed.all_reduce(
            ok, op=torch.distributed.ReduceOp.MIN, group=get_tp_group().cpu_group
        )
        if why is None and not ok.item():
            return "off on another TP rank"
        return why

    def _max_tokens(self) -> int:
        return max(
            self.k1_rows if self.k1 else 0,
            self.k2_rows if self.whole_step or self.k2 else 0,
        )

    @staticmethod
    def _check_widths(cap: int) -> list[int]:
        return sorted({1, min(8, cap), cap})

    def _shuffle_weights(self) -> None:
        from vllm.models.qwen3_5.amd.mono import dense_ffn as D

        # process_weights_after_loading runs after this and rewrites the
        # layers' scales (swizzled for the ASM GEMM, else transposed); the
        # kernels take the checkpoint layout, so keep copies.
        self.w13, self.w2, self.w_o, self.w_in, self.ws = [], [], [], [], []
        for i, layer in enumerate(self.model.layers):
            ws = {}
            self.ws.append(ws)
            if self.whole_step or self.k2:
                self.w13.append(D.shuffle_w(layer.mlp.gate_up_proj.weight.data))
                self.w2.append(D.shuffle_w(layer.mlp.down_proj.weight.data))
                ws["gate_up"] = layer.mlp.gate_up_proj.weight_scale.data.clone()
                ws["down"] = layer.mlp.down_proj.weight_scale.data.clone()
            if self.oproj:
                proj = _o_proj(layer)
                self.w_o.append(D.shuffle_w(proj.weight.data))
                ws["o"] = proj.weight_scale.data.clone()
            if self.k1_oproj:
                proj = layer.linear_attn.out_proj if _is_gdn(layer) else None
                self.w_o.append(None if proj is None else D.shuffle_w(proj.weight.data))
                if proj is not None:
                    ws["out_proj"] = proj.weight_scale.data.clone()
            self.w_in.append(None)
            if self.k1 and _is_gdn(layer):
                attn = layer.linear_attn
                self.w_in[-1] = (
                    D.shuffle_w(attn.in_proj_qkvz.weight.data.contiguous()),
                    D.shuffle_w(attn.in_proj_ba.weight.data.contiguous()),
                )
                ws["qkvz"] = attn.in_proj_qkvz.weight_scale.data.clone()
                ws["ba"] = attn.in_proj_ba.weight_scale.data.clone()
    def _wrap_o(self, proj) -> None:
        stock = proj.forward

        def forward(x, *args, **kwargs):
            return _oproj_forward(self, stock, x, *args, **kwargs)

        proj.forward = forward

    def _wrap_linear_attn(self, index, layer) -> None:
        stock = layer.linear_attn.forward
        self._stock_gdn[index] = stock

        def forward(hidden_states):
            return _gdn_forward(self, index, layer, stock, hidden_states)

        layer.linear_attn.forward = forward

    def _wrap_mlp(self, index, layer) -> None:
        stock = layer.mlp.forward
        self._stock_mlp[index] = stock

        def forward(x):
            return _mlp_forward(self, index, layer, stock, x)

        layer.mlp.forward = forward

    def _k1_from_normed(self, index, layer, x):
        conv, ssm, idx = self._states(layer.linear_attn, x.size(0))
        core, _, out = self._k1(index, layer, conv, ssm, idx, x, None)
        return core, out

    def _states(self, attn, rows: int):
        from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
        from vllm.models.qwen3_5.amd.mono import dense_gdn as K

        meta = self._metadata(attn)
        if meta is None or attn.kv_cache is None or len(attn.kv_cache) < 2:
            raise RuntimeError(
                "VLLM_ROCM_MONO_DECODE K1 has no GDN state for this decode step"
            )
        idx = meta.non_spec_state_indices_tensor
        if idx is None or idx.dtype != torch.int32 or idx.numel() < rows:
            raise RuntimeError(
                "VLLM_ROCM_MONO_DECODE K1 state indices do not cover this step"
            )
        buf = self.idx_buf[:rows]
        buf.copy_(idx[:rows])
        # 0 is vLLM's null block. The kernel leaves a negative slot untouched.
        buf.masked_fill_(buf == 0, -1)
        idx = buf
        conv = attn.kv_cache[0]
        if not is_conv_state_dim_first():
            conv = conv.transpose(-1, -2)
        ssm = attn.kv_cache[1]
        geo = K.Geo(self.tp)
        if conv.dtype != torch.bfloat16 or conv.shape[1:] != (geo.conv, K.SL):
            raise RuntimeError(
                f"VLLM_ROCM_MONO_DECODE K1 conv state {tuple(conv.shape)} "
                f"{conv.dtype}, want [slots, {geo.conv}, {K.SL}] bf16"
            )
        if (
            ssm.dtype != torch.float32
            or ssm.shape[1:] != (geo.nv, K.HD, K.HD)
            or not ssm[0].is_contiguous()
        ):
            raise RuntimeError(
                f"VLLM_ROCM_MONO_DECODE K1 SSM state {tuple(ssm.shape)} {ssm.dtype}"
            )
        return conv, ssm, idx

    def _metadata(self, attn):
        if not is_forward_context_available():
            return None
        raw = get_forward_context().attn_metadata
        if raw is None:
            return None
        if isinstance(raw, dict):
            raw = raw.get(attn.prefix)
        from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

        if not isinstance(raw, GDNAttentionMetadata):
            return None
        return raw

    def _self_check(self) -> str | None:
        why = self._ffn_check() if self.whole_step or self.k2 else None
        if why is None and self.k1:
            why = self._k1_check()
        if why is not None:
            return f"self-check failed: {why}"
        return None

    def _ffn_check(self) -> str | None:
        layer = self.model.layers[0]
        dev = self.epoch.device
        g = torch.Generator(device=dev).manual_seed(0)
        eps = layer.post_attention_layernorm.variance_epsilon
        for s in self._check_widths(self.k2_rows):
            h = torch.randn(s, 5120, generator=g, device=dev).mul_(0.5).bfloat16()
            r = torch.randn(s, 5120, generator=g, device=dev).mul_(2).bfloat16()
            core = torch.randn(s, 6144, generator=g, device=dev).mul_(0.5).bfloat16()
            self.epoch.add_(1)
            if not self.whole_step:
                x, _ = _gemma(h, r, layer.post_attention_layernorm.weight, eps)
                ref = _ref_linear(
                    _ref_silu_and_mul(
                        _ref_linear(
                            x,
                            layer.mlp.gate_up_proj.weight,
                            layer.mlp.gate_up_proj.weight_scale,
                        )
                    ),
                    layer.mlp.down_proj.weight,
                    layer.mlp.down_proj.weight_scale,
                )
                out = self._ffn_partial(0, layer, x)
                if not torch.equal(out, ref):
                    err = (out.float() - ref.float()).abs().max().item()
                    return f"K2 partial at width {s}: max abs err {err}"
                continue
            proj = _o_proj(layer)
            hidden = _ref_linear(core, proj.weight, proj.weight_scale) if self.oproj else h
            x, res_ref = _gemma(hidden, r, layer.post_attention_layernorm.weight, eps)
            ref = _ref_linear(
                _ref_silu_and_mul(
                    _ref_linear(
                        x,
                        layer.mlp.gate_up_proj.weight,
                        layer.mlp.gate_up_proj.weight_scale,
                    )
                ),
                layer.mlp.down_proj.weight,
                layer.mlp.down_proj.weight_scale,
            )
            out, res = self._ffn(0, layer, core if self.oproj else h, r)
            if not torch.equal(res, res_ref):
                return f"K2 res_out at width {s}"
            if not torch.equal(out, ref):
                err = (out.float() - ref.float()).abs().max().item()
                return f"K2 out at width {s}: max abs err {err}"
        return None

    def _k1_check(self) -> str | None:
        from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
            causal_conv1d_update,
        )
        from vllm.models.qwen3_5.amd.mono import dense_gdn as K
        from vllm.third_party.flash_linear_attention.ops import (
            fused_recurrent_gated_delta_rule_packed_decode,
        )

        geo = K.Geo(self.tp)
        index, layer = next(
            (i, lyr) for i, lyr in enumerate(self.model.layers) if _is_gdn(lyr)
        )
        attn = layer.linear_attn
        dev = self.epoch.device
        g = torch.Generator(device=dev).manual_seed(1)
        eps = layer.input_layernorm.variance_epsilon
        for s in self._check_widths(self.k1_rows):
            first = s == 1
            h = torch.randn(s, K.HIDDEN, generator=g, device=dev).mul_(0.5).bfloat16()
            residual = (
                None
                if first
                else torch.randn(s, K.HIDDEN, generator=g, device=dev).mul_(2).bfloat16()
            )
            conv = torch.randn(1 + s, geo.conv, K.SL, generator=g, device=dev).bfloat16()
            # vLLM loads weights under a bf16 default dtype. The SSM cache and
            # the kernel are fp32; a bf16 draw fails the launcher assert.
            ssm = torch.randn(
                1 + s,
                geo.nv,
                K.HD,
                K.HD,
                generator=g,
                device=dev,
                dtype=torch.float32,
            ).mul_(0.05)
            idx = torch.arange(1, 1 + s, dtype=torch.int32, device=dev)
            if s > 1:
                idx[1] = -1
            conv_ref, ssm_ref = conv.clone(), ssm.clone()
            conv_f32, ssm_f32 = conv.float(), ssm.clone()
            x, res_ref = _gemma(h, residual, layer.input_layernorm.weight, eps)
            core_f32, out_f32 = _k1_f32(attn, geo, x, conv_f32, ssm_f32, idx)
            qkv_size = (attn.key_dim * 2 + attn.value_dim) // attn.tp_size
            z_size = attn.value_dim // attn.tp_size
            mixed_qkvz = _ref_linear(
                x, attn.in_proj_qkvz.weight, attn.in_proj_qkvz.weight_scale
            )
            ba = _ref_linear(x, attn.in_proj_ba.weight, attn.in_proj_ba.weight_scale)
            mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)
            b, a = ba.chunk(2, dim=-1)
            mixed = causal_conv1d_update(
                mixed_qkv,
                conv_ref,
                attn.conv1d.weight.view(attn.conv1d.weight.size(0), -1),
                None,
                attn.activation,
                conv_state_indices=idx,
                null_block_id=-1,
            )
            o = mixed.new_empty(s, 1, geo.nv, attn.head_v_dim)
            fused_recurrent_gated_delta_rule_packed_decode(
                mixed_qkv=mixed,
                a=a.contiguous(),
                b=b.contiguous(),
                A_log=attn.A_log,
                dt_bias=attn.dt_bias,
                scale=attn.head_k_dim**-0.5,
                initial_state=ssm_ref,
                out=o,
                ssm_state_indices=idx,
                use_qk_l2norm_in_kernel=True,
            )
            core_ref = attn.norm(o.view(-1, attn.head_v_dim), z.reshape(-1, attn.head_v_dim))
            core_ref = core_ref.view(s, -1)
            self.epoch.add_(1)
            out = out_ref = None
            if self.whole_step:
                core, res, _ = self._k1(index, layer, conv, ssm, idx, h, residual)
            else:
                core, res, out = self._k1(index, layer, conv, ssm, idx, x, None)
                res = res_ref
                if out is not None:
                    out_ref = _ref_linear(
                        core_ref, attn.out_proj.weight, attn.out_proj.weight_scale
                    )
            why = _k1_verdict(
                s,
                idx >= 0,
                (res, res_ref),
                {
                    "core": (core, core_ref, core_f32),
                    "out_proj partial": (out, out_ref, out_f32),
                    "conv state": (conv, conv_ref, conv_f32),
                    "SSM state": (ssm, ssm_ref, ssm_f32),
                },
            )
            if why is not None:
                return why
        return None

    def _pure_decode(self, rows: int) -> bool:
        if not self._prepared or not 1 <= rows <= self._max_tokens():
            return False
        if self.model.aux_hidden_state_layers:
            return self._skip("aux hidden states")
        attn = next(
            (layer.linear_attn for layer in self.model.layers if _is_gdn(layer)),
            None,
        )
        if attn is None:
            return False
        meta = self._metadata(attn)
        if meta is None:
            return False
        if meta.num_prefills != 0 or meta.num_decodes <= 0 or meta.spec_sequence_masks is not None:
            return False
        if meta.num_spec_decodes:
            return self._skip("spec decode")
        return True

    def _skip(self, why: str) -> bool:
        if why not in self._logged:
            self._logged.add(why)
            logger.warning("Qwen3.8 dense mono skips a decode step: %s", why)
        return False

    def begin_step(self, rows: int) -> None:
        """TP2, before the stock layers: whether this step's GDN mixers take K1
        and its FFNs take K2."""
        if not self._prepared and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "VLLM_ROCM_MONO_DECODE was first reached inside a CUDA graph "
                "capture, before an eager decode step."
            )
        on = self._pure_decode(rows)
        self._gdn_step = on and self.k1 and rows <= self.k1_rows
        self._ffn_step = on and self.k2 and rows <= self.k2_rows
        if self._gdn_step or self._ffn_step:
            self.epoch.add_(1)
            if rows not in self._logged:
                self._logged.add(rows)
                logger.info(
                    "Qwen3.8 dense mono step at width %d (K1 %s, K2 %s)",
                    rows,
                    "on" if self._gdn_step else "off",
                    "on" if self._ffn_step else "off",
                )

    def try_whole_forward(
        self,
        input_ids,
        positions,
        intermediate_tensors,
        inputs_embeds,
    ):
        """Returns the hidden states when this step is pure decode within a
        kernel cap, else None so the compiled stock model runs."""
        if self.graph is None or intermediate_tensors is not None:
            return None
        rows = positions.shape[-1]
        if not self._pure_decode(rows):
            return None
        if not self.whole_step:
            # TP1 counts the step inside the first layer op.
            self.epoch.add_(1)
        if rows not in self._logged:
            self._logged.add(rows)
            logger.info("Qwen3.8 dense mono step at width %d", rows)
        return self.graph(input_ids, positions, inputs_embeds)

    def _layer(self, index, hidden, residual, positions):
        """TP1 whole-layer K1/K2 path used by the opaque per-layer op."""
        rows = positions.shape[-1]
        if index == 0:
            if rows not in self._logged:
                self._logged.add(rows)
                logger.info("Qwen3.8 dense mono layer graph at width %d", rows)
            self.epoch.add_(1)
        layer = self.model.layers[index]
        k2 = rows <= self.k2_rows
        if self.k1 and rows <= self.k1_rows and _is_gdn(layer):
            conv, ssm, idx = self._states(layer.linear_attn, rows)
            core, res, _ = self._k1(index, layer, conv, ssm, idx, hidden, residual)
            if not k2:
                core, _ = layer.linear_attn.out_proj(core)
            return self._tail(index, layer, core, res, k2)
        if residual is None:
            res = hidden
            x = layer.input_layernorm(hidden)
        else:
            x, res = layer.input_layernorm(hidden, residual)
        self._skip_o = self.oproj and k2
        try:
            mixed = (
                layer.linear_attn(x)
                if _is_gdn(layer)
                else layer.self_attn(positions=positions, hidden_states=x)
            )
        finally:
            self._skip_o = False
        return self._tail(index, layer, mixed, res, k2)

    def _layers(self, hidden, positions, k2):
        residual = None
        k1 = self.k1 and hidden.size(0) <= self.k1_rows
        for i, layer in enumerate(self.model.layers):
            if k1 and _is_gdn(layer):
                conv, ssm, idx = self._states(layer.linear_attn, hidden.size(0))
                core, res, _ = self._k1(i, layer, conv, ssm, idx, hidden, residual)
                if not k2:
                    core, _ = layer.linear_attn.out_proj(core)
                hidden, residual = self._tail(i, layer, core, res, k2)
                continue
            if residual is None:
                res = hidden
                x = layer.input_layernorm(hidden)
            else:
                x, res = layer.input_layernorm(hidden, residual)
            if _is_gdn(layer):
                mixed = layer.linear_attn(x)
            else:
                mixed = layer.self_attn(positions=positions, hidden_states=x)
            hidden, residual = self._tail(i, layer, mixed, res, k2)
        return hidden, residual

    def _tail(self, index, layer, hidden, residual, k2):
        if k2:
            return self._ffn(index, layer, hidden, residual)
        x, residual = layer.post_attention_layernorm(hidden, residual)
        return layer.mlp(x), residual

    def _k1(self, index, layer, conv_state, ssm_state, idx, hidden, residual):
        from vllm.models.qwen3_5.amd.mono import dense_gdn

        attn = layer.linear_attn
        rows = hidden.size(0)
        key = self._k1_key(
            dense_gdn,
            rows,
            first=residual is None,
            eps=layer.input_layernorm.variance_epsilon,
            norm_eps=attn.norm.eps,
        )
        core = torch.empty(
            rows, dense_gdn.Geo(self.tp).val, dtype=hidden.dtype, device=hidden.device
        )
        front = {"hidden": hidden.contiguous()}
        res_out = out = None
        if not key.normed:
            res_out = torch.empty_like(hidden)
            front.update(
                residual=residual,
                res_out=res_out,
                ln_w=layer.input_layernorm.weight,
            )
        if key.oproj:
            out = torch.empty_like(hidden)
            front.update(
                w_o=self.w_o[index],
                w_os=self.ws[index]["out_proj"],
                out=out,
            )
        w_qkvz, w_ba = self.w_in[index]
        dense_gdn.dense_gdn_pre(
            key,
            **front,
            w_qkvz=w_qkvz,
            w_qkvzs=self.ws[index]["qkvz"],
            w_ba=w_ba,
            w_bas=self.ws[index]["ba"],
            conv_w=attn.conv1d.weight,
            conv_state=conv_state,
            a_log=attn.A_log,
            dt_bias=attn.dt_bias,
            norm_w=attn.norm.weight,
            rstate=ssm_state,
            st_idx=idx,
            core=core,
            scratch=self.k1_scratch,
            epoch=self.epoch,
            layer=index,
        )
        return core, res_out, out

    def _ffn(self, index, layer, hidden, residual):
        from vllm.models.qwen3_5.amd.mono import dense_ffn as D

        out = torch.empty_like(residual)
        res_out = torch.empty_like(residual)
        D.dense_ffn(
            self._k2_key(D, hidden.size(0), eps=layer.post_attention_layernorm.variance_epsilon),
            hidden=hidden,
            w_o=self.w_o[index] if self.oproj else None,
            w_os=self.ws[index]["o"] if self.oproj else None,
            residual=residual,
            ln_w=layer.post_attention_layernorm.weight,
            w13=self.w13[index],
            w13s=self.ws[index]["gate_up"],
            w2=self.w2[index],
            w2s=self.ws[index]["down"],
            out=out,
            res_out=res_out,
            scratch=self.scratch,
            epoch=self.epoch,
            layer=index,
        )
        return out, res_out

    def _ffn_partial(self, index, layer, x):
        from vllm.models.qwen3_5.amd.mono import dense_ffn as D

        out = torch.empty_like(x)
        D.dense_ffn(
            self._k2_key(D, x.size(0)),
            hidden=x.contiguous(),
            residual=None,
            ln_w=None,
            w13=self.w13[index],
            w13s=self.ws[index]["gate_up"],
            w2=self.w2[index],
            w2s=self.ws[index]["down"],
            out=out,
            res_out=None,
            scratch=self.scratch,
            epoch=self.epoch,
            layer=index,
        )
        return out


def install(model, vllm_config) -> DenseMono | None:
    """Construct the dense path when the flag is set on ``qwen3_5_text``.

    The 2.4T MoE class stays on the stock path: this kernel does not run it.
    A dense model the kernels cannot run raises.
    """
    if not envs.VLLM_ROCM_MONO_DECODE:
        return None
    if model.config.model_type != "qwen3_5_text":
        logger.info(
            "VLLM_ROCM_MONO_DECODE left %s on the stock path "
            "(dense mono decode is the Qwen3.8-27B text model)",
            model.config.model_type,
        )
        return None
    return DenseMono(model, vllm_config)

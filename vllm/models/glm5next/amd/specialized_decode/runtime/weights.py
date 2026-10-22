# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One GLM-5.3-Flash MoE layer's tensors, as views of vLLM's loaded parameters
(the router and a shared expert stored apart from the routed ones excepted: they
are packed once into the kernel's layouts)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vllm.models.glm5next.amd.specialized_decode.mono.config import (
    HC_MULT,
    HIDDEN,
    N_EXPERTS,
)

if TYPE_CHECKING:
    from torch import nn

    from vllm.models.glm5next.amd.specialized_decode.mono.ffn.op import (
        FlashDsaWeights,
        FlashFfnWeights,
        FlashKdaWeights,
    )

N_MIX = HC_MULT * (2 + HC_MULT)


def _need(
    name: str, t: torch.Tensor, shape: tuple, dtypes: tuple[torch.dtype, ...]
) -> torch.Tensor:
    if tuple(t.shape) != shape or t.dtype not in dtypes or not t.is_contiguous():
        raise ValueError(
            f"needs {name} as contiguous {'/'.join(map(str, dtypes))} {shape}, "
            f"got {t.dtype} {tuple(t.shape)}"
        )
    return t


def _bytes(name: str, t: torch.Tensor, shape: tuple) -> torch.Tensor:
    """An MXFP4 / E8M0 tensor viewed as raw bytes, checked."""
    if t.element_size() != 1 or tuple(t.shape) != shape or not t.is_contiguous():
        raise ValueError(
            f"needs {name} as contiguous 1-byte {shape}, got {t.dtype} {tuple(t.shape)}"
        )
    return t.view(torch.uint8)


def experts_backend_reason(layer: nn.Module) -> str | None:
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import Mxfp4MoeBackend

    mlp = layer.mlp
    fused = getattr(mlp, "is_fused_shared_expert_enabled", False)
    if not fused and getattr(mlp, "shared_experts", None) is None:
        return "needs a shared expert"
    routed = getattr(getattr(mlp, "experts", None), "routed_experts", None)
    if routed is None:
        return "needs a routed MoE"
    backend = getattr(routed.quant_method, "mxfp4_backend", None)
    if backend != Mxfp4MoeBackend.AITER_MXFP4_MXFP4:
        return f"needs the AITER_MXFP4_MXFP4 MoE backend, not {backend}"
    return None


def _mxfp4_rows(name: str, linear: nn.Module, rows: int, k: int):
    """A Quark MXFP4 linear in the AITER Triton GEMM layout (weight ``[rows, K/2]``,
    scale stored ``[K/32, rows]``); returns the weight and a ``[rows, K/32]`` scale."""
    w, sc = linear.weight, getattr(linear, "weight_scale", None)
    if (
        sc is None
        or w.element_size() != 1
        or sc.element_size() != 1
        or tuple(w.shape) != (rows, k // 2)
        or tuple(sc.shape) != (k // 32, rows)
    ):
        got = None if sc is None else tuple(sc.shape)
        raise ValueError(
            f"needs shared_experts.{name} as row-major MXFP4 {(rows, k // 2)} with a "
            f"{(k // 32, rows)} scale (VLLM_ROCM_USE_AITER_FP4_ASM_GEMM off), got "
            f"{w.dtype} {tuple(w.shape)} and scale {got}"
        )
    return w.view(torch.uint8), sc.view(torch.uint8).T.contiguous()


def _shared_expert(mlp: nn.Module, inter: int):
    from vllm.models.glm5next.amd.specialized_decode.mono.packing import (
        pack_shared_expert,
    )

    gate_up = _mxfp4_rows("gate_up_proj", mlp.gate_up_proj, 2 * inter, HIDDEN)
    down = _mxfp4_rows("down_proj", mlp.down_proj, HIDDEN, inter)
    return pack_shared_expert(*gate_up, *down)


def layer_weights(layer: nn.Module, inter: int) -> FlashFfnWeights:
    from vllm.models.glm5next.amd.specialized_decode.mono.ffn.op import FlashFfnWeights
    from vllm.models.glm5next.amd.specialized_decode.mono.packing import pack_bf16

    why = experts_backend_reason(layer)
    if why is not None:
        raise ValueError(why)
    mlp = layer.mlp
    routed = mlp.experts.routed_experts
    fused = mlp.is_fused_shared_expert_enabled
    E = N_EXPERTS + 1 if fused else N_EXPERTS
    w_ug = _bytes("w13_weight", routed.w13_weight, (E, 2 * inter, HIDDEN // 2))
    s_ug = _bytes(
        "w13_weight_scale", routed.w13_weight_scale, (E, 2 * inter, HIDDEN // 32)
    )
    w_dn = _bytes("w2_weight", routed.w2_weight, (E, HIDDEN, inter // 2))
    s_dn = _bytes("w2_weight_scale", routed.w2_weight_scale, (E, HIDDEN, inter // 32))
    for name, t in (("w13_weight", routed.w13_weight), ("w2_weight", routed.w2_weight)):
        if not getattr(t, "is_shuffled", False):
            raise ValueError(f"needs {name} in AITER's shuffled layout")
    if fused:
        shared = (w_ug[N_EXPERTS], s_ug[N_EXPERTS], w_dn[N_EXPERTS], s_dn[N_EXPERTS])
    else:
        shared = _shared_expert(mlp.shared_experts, inter)
    gate = _need(
        "gate.weight",
        mlp.gate.weight,
        (N_EXPERTS, HIDDEN),
        (torch.float32, torch.bfloat16),
    )
    router = gate.to(torch.bfloat16)
    if gate.dtype != torch.bfloat16 and not torch.equal(router.float(), gate):
        raise ValueError("needs router weights exactly representable in bf16")
    bias = _need(
        "gate.e_score_correction_bias",
        mlp.gate.e_score_correction_bias,
        (N_EXPERTS,),
        (torch.float32,),
    )
    return FlashFfnWeights(
        w_r=pack_bf16(router),
        bias=bias,
        w_ug=w_ug,
        s_ug=s_ug,
        w_dn=w_dn,
        s_dn=s_dn,
        w_sh_ug=shared[0],
        s_sh_ug=shared[1],
        w_sh_dn=shared[2],
        s_sh_dn=shared[3],
        hc_fn=_need(
            "hc_ffn_fn", layer.hc_ffn_fn, (N_MIX, HC_MULT * HIDDEN), (torch.float32,)
        ),
        hc_scale=_need("hc_ffn_scale", layer.hc_ffn_scale, (3,), (torch.float32,)),
        hc_base=_need("hc_ffn_base", layer.hc_ffn_base, (N_MIX,), (torch.float32,)),
        g_norm=_need(
            "post_attention_layernorm.weight",
            layer.post_attention_layernorm.weight,
            (HIDDEN,),
            (torch.bfloat16,),
        ),
    )


def _proj(name: str, linear: nn.Module, rows: int, cols: int):
    """A row-major projection as vLLM keeps it after loading: FP8 E4M3 with f32
    128 x 128 block scales, or bf16 (scale None)."""
    w = linear.weight
    sc = getattr(linear, "weight_scale_inv", None)
    if sc is None:
        sc = getattr(linear, "weight_scale", None)
    if w.dtype == torch.bfloat16 and tuple(w.shape) == (rows, cols):
        return _need(f"{name}.weight", w, (rows, cols), (torch.bfloat16,)), None
    if (
        w.dtype != torch.float8_e4m3fn
        or tuple(w.shape) != (rows, cols)
        or not w.is_contiguous()
        or sc is None
        or sc.dtype != torch.float32
        or tuple(sc.shape) != (rows // 128, cols // 128)
        or not sc.is_contiguous()
    ):
        got = None if sc is None else f"{sc.dtype} {tuple(sc.shape)}"
        raise ValueError(
            f"needs {name} as row-major bf16 {(rows, cols)}, or FP8 E4M3 with f32 "
            f"{(rows // 128, cols // 128)} block scales; got {w.dtype} "
            f"{tuple(w.shape)} and scale {got}"
        )
    return w, sc


def dsa_weights(layer: nn.Module, tp: int) -> FlashDsaWeights:
    """A sparse-attention layer's attention side as views of vLLM's parameters;
    W_UK^T is packed once from kv_b_proj into [H, 512, 256] bf16."""
    from vllm.models.glm5next.amd.specialized_decode.mono.config import (
        INDEX_DIM,
        INDEX_HEADS,
        KPOOL,
        KV_LORA,
        NOPE_DIM,
        Q_LORA,
        V_DIM,
        FlashShard,
    )
    from vllm.models.glm5next.amd.specialized_decode.mono.ffn.op import FlashDsaWeights

    attn = layer.self_attn
    idx = attn.indexer
    H = FlashShard(tp).mla_heads
    bf16, f32 = (torch.bfloat16,), (torch.float32,)
    w_qkv, s_qkv = _proj(
        "fused_qkv_a_proj", attn.fused_qkv_a_proj, Q_LORA + KV_LORA, HIDDEN
    )
    w_qb, s_qb = _proj("q_b_proj", attn.q_b_proj, H * NOPE_DIM, Q_LORA)
    w_o, s_o = _proj("o_proj", attn.o_proj, HIDDEN, H * V_DIM)
    per = NOPE_DIM + V_DIM
    kv_b = _need("kv_b_proj.weight", attn.kv_b_proj.weight, (H * per, KV_LORA), bf16)
    w_uk = kv_b.view(H, per, KV_LORA)[:, :NOPE_DIM].transpose(1, 2).contiguous()
    return FlashDsaWeights(
        hc_fn=_need("hc_attn_fn", layer.hc_attn_fn, (N_MIX, HC_MULT * HIDDEN), f32),
        hc_scale=_need("hc_attn_scale", layer.hc_attn_scale, (3,), f32),
        hc_base=_need("hc_attn_base", layer.hc_attn_base, (N_MIX,), f32),
        g_norm=_need(
            "input_layernorm.weight", layer.input_layernorm.weight, (HIDDEN,), bf16
        ),
        w_qkv=w_qkv,
        s_qkv=s_qkv,
        w_idx=_need(
            "indexer.wk_weights_proj.weight",
            idx.wk_weights_proj.weight,
            (INDEX_DIM + INDEX_HEADS, HIDDEN),
            bf16,
        ),
        w_gate=_need(
            "indexer.index_kpool_compress_gate",
            idx.index_kpool_compress_gate,
            (INDEX_DIM, HIDDEN),
            bf16,
        ),
        g_qa=_need("q_a_layernorm.weight", attn.q_a_layernorm.weight, (Q_LORA,), bf16),
        g_kv=_need(
            "kv_a_layernorm.weight", attn.kv_a_layernorm.weight, (KV_LORA,), bf16
        ),
        ln_w=_need("indexer.k_norm.weight", idx.k_norm.weight, (INDEX_DIM,), f32),
        ln_b=_need("indexer.k_norm.bias", idx.k_norm.bias, (INDEX_DIM,), f32),
        ape=_need(
            "indexer.index_kpool_compress_ape",
            idx.index_kpool_compress_ape,
            (KPOOL, INDEX_DIM),
            f32,
        ),
        w_qb=w_qb,
        s_qb=s_qb,
        w_iq=_need(
            "indexer.wq_b.weight",
            idx.wq_b.weight,
            (INDEX_HEADS * INDEX_DIM, Q_LORA),
            bf16,
        ),
        w_uk=w_uk,
        w_kvb=kv_b,
        w_o=w_o,
        s_o=s_o,
    )


def kda_weights(layer: nn.Module, tp: int) -> FlashKdaWeights:
    """A KDA layer's attention side, as views of vLLM's row-major bf16 projections;
    the q | k | v conv taps are packed once into one [3P, 4] tensor."""
    from vllm.models.glm5next.amd.specialized_decode.mono.config import (
        KDA_DIM,
        FlashShard,
    )
    from vllm.models.glm5next.amd.specialized_decode.mono.ffn.op import FlashKdaWeights

    attn = layer.self_attn
    shard = FlashShard(tp)
    P, H, D = shard.kda_proj, shard.kda_heads, KDA_DIM
    bf16, f32 = (torch.bfloat16,), (torch.float32,)

    def linear(name, rows, cols):
        return _need(f"{name}.weight", getattr(attn, name).weight, (rows, cols), bf16)

    taps = []
    for name in ("q_conv1d", "k_conv1d", "v_conv1d"):
        conv = getattr(attn, name)
        if conv.bias is not None:
            raise ValueError(f"needs {name} without bias")
        taps.append(_need(f"{name}.weight", conv.weight, (P, 1, 4), f32).view(P, 4))
    w_on = attn.o_norm.weight
    if w_on is None or tuple(w_on.shape) != (D,):
        raise ValueError("needs o_norm with a weight")
    return FlashKdaWeights(
        hc_fn=_need("hc_attn_fn", layer.hc_attn_fn, (N_MIX, HC_MULT * HIDDEN), f32),
        hc_scale=_need("hc_attn_scale", layer.hc_attn_scale, (3,), f32),
        hc_base=_need("hc_attn_base", layer.hc_attn_base, (N_MIX,), f32),
        g_norm=_need(
            "input_layernorm.weight", layer.input_layernorm.weight, (HIDDEN,), bf16
        ),
        w_in=linear("in_proj_qkvbfg_a", shard.kda_in_rows, HIDDEN),
        w_fb=linear("f_b_proj", P, D),
        w_gb=linear("g_b_proj", P, D),
        conv_w=torch.cat(taps).contiguous(),
        a_log=_need("A_log", attn.A_log, (1, 1, H, 1), f32).view(H),
        dt_bias=_need("dt_bias", attn.dt_bias, (P,), f32),
        w_on=w_on.float().contiguous(),
        w_o=linear("o_proj", HIDDEN, P),
    )

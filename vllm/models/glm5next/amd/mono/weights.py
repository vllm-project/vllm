# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One GLM-5.3-Flash MoE layer's tensors, as views of vLLM's loaded parameters
(the router excepted: it is packed once into the kernel's bf16 tiles)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vllm.models.glm5next.amd.ops.mono.config import (
    HC_MULT,
    HIDDEN,
    N_EXPERTS,
)

if TYPE_CHECKING:
    from torch import nn

    from vllm.models.glm5next.amd.ops.mono.ffn.op import FlashFfnWeights

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
    if not getattr(mlp, "is_fused_shared_expert_enabled", False):
        return "needs VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1"
    routed = getattr(getattr(mlp, "experts", None), "routed_experts", None)
    if routed is None:
        return "needs a routed MoE"
    backend = getattr(routed.quant_method, "mxfp4_backend", None)
    if backend != Mxfp4MoeBackend.AITER_MXFP4_MXFP4:
        return f"needs the AITER_MXFP4_MXFP4 MoE backend, not {backend}"
    return None


def layer_weights(layer: nn.Module, inter: int) -> FlashFfnWeights:
    from vllm.models.glm5next.amd.ops.mono.ffn.op import FlashFfnWeights
    from vllm.models.glm5next.amd.ops.mono.packing import pack_bf16

    why = experts_backend_reason(layer)
    if why is not None:
        raise ValueError(why)
    mlp = layer.mlp
    routed = mlp.experts.routed_experts
    E = N_EXPERTS + 1
    w_ug = _bytes("w13_weight", routed.w13_weight, (E, 2 * inter, HIDDEN // 2))
    s_ug = _bytes(
        "w13_weight_scale", routed.w13_weight_scale, (E, 2 * inter, HIDDEN // 32)
    )
    w_dn = _bytes("w2_weight", routed.w2_weight, (E, HIDDEN, inter // 2))
    s_dn = _bytes("w2_weight_scale", routed.w2_weight_scale, (E, HIDDEN, inter // 32))
    for name, t in (("w13_weight", routed.w13_weight), ("w2_weight", routed.w2_weight)):
        if not getattr(t, "is_shuffled", False):
            raise ValueError(f"needs {name} in AITER's shuffled layout")
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

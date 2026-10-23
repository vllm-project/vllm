# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The gfx942 kernels' own copies of a decoder layer's weights.

On CDNA3 the kernels compute the same layer with FNUZ FP8 MFMAs, from tensors
re-encoded as they read them (``mono/weights942.py``). The copies are made as
vLLM processes the loaded weights, so that they count as model memory when vLLM
sizes the KV cache: copies made at the first decode step would count only when
that step is also the CUDA graph memory profile.
"""

from typing import TYPE_CHECKING

import torch

from ..spec import INTER_TP4

if TYPE_CHECKING:
    from vllm.models.deepseek_v41.amd.model import DeepseekV4DecoderLayer

# the attention linears the kernels read as ``linear_copy`` copies, as names of
# the layer's attention module
ATTN_LINEARS = ("fused_wqa_wkv", "wq_b", "wo_a", "wo_b")


def scale_bytes(scale: torch.Tensor) -> torch.Tensor:
    return scale if scale.dtype == torch.uint8 else scale.view(torch.uint8)


def _keep(layer, copies: tuple[torch.Tensor, ...]) -> None:
    """Store a layer's copies as ``layer.mono942``. A reload of the weights
    runs ``process_weights_after_loading`` again, and the kernels and captured
    CUDA graphs read the first copies by their addresses, so a reload's copies
    are written into the first ones in place."""
    kept = getattr(layer, "mono942", None)
    if kept is None:
        layer.mono942 = copies
        return
    for old, new in zip(kept, copies, strict=True):
        old.copy_(new)


def _capture_plain_experts(experts) -> None:
    """Take the copies before vLLM converts the routed experts to its Triton kernels'
    format when the weights are loaded, which frees the plain scales."""
    qm = experts.quant_method
    original = qm.process_weights_after_loading

    def process_weights_after_loading(layer):
        if layer is experts:
            from ...mono.weights942 import moe_copies

            _keep(
                layer,
                moe_copies(
                    layer.w13_weight.data,
                    scale_bytes(layer.w13_weight_scale.data),
                    layer.w2_weight.data,
                    scale_bytes(layer.w2_weight_scale.data),
                    INTER_TP4,
                ),
            )
        return original(layer)

    qm.process_weights_after_loading = process_weights_after_loading


def _copy_linear(linear) -> None:
    """By default (VLLM_MXFP8_EMULATION_DEQUANT_AT_LOAD) vLLM dequantizes a
    dense MXFP8 linear's weights to bf16, and the copy re-encodes them as the
    kernels read them. Weights that stay e4m3 are copied as they are."""
    qm = linear.quant_method
    original = qm.process_weights_after_loading

    def process_weights_after_loading(layer):
        out = original(layer)
        if layer is linear:
            from ...mono.weights942 import linear_copy

            _keep(layer, linear_copy(layer.weight, layer.weight_scale))
        return out

    qm.process_weights_after_loading = process_weights_after_loading


def copy_at_load(layer: "DeepseekV4DecoderLayer", ffn_only: bool) -> None:
    """Make this layer's copies when vLLM processes its weights at load. A
    layer that takes only the FFN launch needs no attention copies."""
    ffn = layer.ffn
    _capture_plain_experts(ffn.experts.routed_experts)
    sh = ffn.shared_experts
    assert sh is not None
    linears = [sh.gate_up_proj, sh.down_proj]
    if not ffn_only:
        linears += [getattr(layer.attn, name) for name in ATTN_LINEARS]
    for linear in linears:
        _copy_linear(linear)


def layer_weights(layer: "DeepseekV4DecoderLayer", ffn_only: bool, world: int):
    """The layer's tensors as the gfx942 kernels read them: the dense MXFP8
    linears and the routed experts as the copies made at load."""
    from ...mono.runner import MonoLayerWeights

    f = layer.ffn
    e, sh = f.experts.routed_experts, f.shared_experts
    assert sh is not None
    w13, w13_s, w2, w2_s = e.mono942
    sgu, sgu_s = sh.gate_up_proj.mono942
    sw2, sw2_s = sh.down_proj.mono942
    weights = MonoLayerWeights(
        attn=None if ffn_only else _attn_weights(layer.attn, world),
        hc_attn_fn=layer.hc_attn_fn,
        hc_attn_scale=layer.hc_attn_scale,
        hc_attn_base=layer.hc_attn_base,
        attn_norm=layer.attn_norm.weight,
        hc_ffn_fn=layer.hc_ffn_fn,
        hc_ffn_scale=layer.hc_ffn_scale,
        hc_ffn_base=layer.hc_ffn_base,
        ffn_norm=layer.ffn_norm.weight,
        gate_w=f.gate.weight,
        bias=f.gate.e_score_correction_bias,
        w13=w13,
        w13_s=w13_s,
        w2=w2,
        w2_s=w2_s,
        sgu=sgu,
        sgu_s=sgu_s,
        sw2=sw2,
        sw2_s=sw2_s,
    )
    weights.check_942(f.n_routed_experts)
    return weights


def _attn_weights(a, world: int):
    from ...mono.attention.plan import Dims
    from ...mono.runner import AttnWeights

    wqkv, wq_b, wo_a, wo_b = (getattr(a, name).mono942 for name in ATTN_LINEARS)
    attn = AttnWeights(
        layer_id=a.layer_id,
        wqkv=wqkv[0],
        wqkv_scale=wqkv[1],
        q_norm=a.q_norm.weight,
        kv_norm=a.kv_norm.weight,
        wq_b=wq_b[0],
        wq_b_scale=wq_b[1],
        wo_a=wo_a[0],
        wo_a_scale=wo_a[1],
        wo_b=wo_b[0],
        wo_b_scale=wo_b[1],
        attn_sink=a.attn_sink,
        cos_sin=a.rotary_emb.cos_sin_cache,
        ratio=a.compress_ratio,
    )
    attn.check(Dims(world))
    return attn

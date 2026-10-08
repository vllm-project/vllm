# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checked, model-owned weight views for the opt-in ATOM mono kernels.

Native parameters stay available for prefill and fallback. Only QKV/O and the
router get conversion copies; fused MXFP4 experts and native cache storage are
shared. Import this module only after mono has been explicitly enabled.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm.models.minimax_m3.amd.model import MiniMaxM3SparseAttention


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"MiniMax-M3 ATOM mono: {message}")


def checked_tensor(
    tensor: torch.Tensor,
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    require(
        tuple(tensor.shape) == shape
        and tensor.dtype == dtype
        and tensor.device == device
        and tensor.is_contiguous()
        and tensor.data_ptr() % 16 == 0,
        f"{name}: expected contiguous aligned {shape} {dtype} on {device}; "
        f"got {tuple(tensor.shape)} {tensor.dtype} {tensor.device} "
        f"stride={tensor.stride()} offset={tensor.storage_offset()}",
    )
    return tensor


def _ptpc(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    from aiter.ops.shuffle import shuffle_weight

    values = weight.float()
    scale = values.abs().amax(dim=1).div(448).clamp_min(1e-30)
    quantized = (values / scale[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return shuffle_weight(quantized, layout=(16, 16)), scale


@dataclass(frozen=True)
class MonoLayerWeights:
    layer_id: int
    attention: "MiniMaxM3SparseAttention"
    g_in: torch.Tensor
    w_qkv: torch.Tensor
    s_qkv: torch.Tensor
    g_q: torch.Tensor
    g_k: torch.Tensor
    g_iq: torch.Tensor
    g_ik: torch.Tensor
    cos_sin: torch.Tensor
    w_o: torch.Tensor
    s_o: torch.Tensor
    g_post: torch.Tensor
    gate: torch.Tensor
    bias: torch.Tensor
    w13: torch.Tensor
    s13: torch.Tensor
    w2: torch.Tensor
    s2: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    index_cache: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor

    @classmethod
    def from_layer(cls, layer) -> "MonoLayerWeights":
        attn = layer.self_attn
        moe = layer.block_sparse_moe
        experts = moe.experts.routed_experts
        device = attn.qkv_proj.weight.device
        label = f"layer {layer.layer_id}"
        require(
            (attn.num_heads, attn.num_kv_heads, attn.num_idx_heads, attn.head_dim)
            == (16, 1, 1, 128),
            f"{label}: unsupported Q/KV/index heads",
        )
        require(
            attn.use_aiter_sparse_pa
            and not attn.skip_index_topk
            and attn.rotary_emb.rotary_dim == 64
            and attn.kv_cache_dtype == "fp8",
            f"{label}: attention/cache/RoPE configuration",
        )
        require(
            moe.is_fused_shared_expert_enabled
            and moe.use_aiter_moe_fse
            and moe.shared_experts is None
            and experts.w13_weight.shape[0] == 129,
            f"{label}: enable VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1",
        )
        require(
            str(experts.quant_method.mxfp4_backend).endswith("AITER_MXFP4_MXFP4")
            and getattr(experts.w13_weight, "is_shuffled", False)
            and getattr(experts.w2_weight, "is_shuffled", False),
            f"{label}: expected shuffled AITER MXFP4 expert weights",
        )

        def check(tensor, name, shape, dtype=torch.bfloat16):
            return checked_tensor(tensor, f"{label} {name}", shape, dtype, device)

        w_qkv, s_qkv = _ptpc(check(attn.qkv_proj.weight, "QKV", (2560, 6144)))
        w_o, s_o = _ptpc(check(attn.o_proj.weight, "O", (6144, 2048)))
        gate = check(moe.gate.weight, "router", (128, 6144), torch.float32).bfloat16()
        cos_sin = attn.rotary_emb.cos_sin_cache
        check(cos_sin, "cos/sin", (cos_sin.shape[0], 64))
        w13 = check(
            experts.w13_weight, "w13", (129, 1536, 3072), torch.float4_e2m1fn_x2
        )
        w2 = check(experts.w2_weight, "w2", (129, 6144, 384), torch.float4_e2m1fn_x2)
        s13 = check(experts.w13_weight_scale, "s13", (129, 1536, 192), torch.uint8)
        s2 = check(experts.w2_weight_scale, "s2", (129, 6144, 24), torch.uint8)
        cache = attn.kv_cache
        require(
            cache.ndim == 4
            and tuple(cache.shape[1:]) == (2, 128, 128)
            and cache.element_size() == 1
            and cache.is_contiguous()
            and cache.device == device,
            f"{label}: native packed K/V layout",
        )
        k, v = attn.get_aiter_sparse_pa_kv_cache()
        require(
            k.dtype == v.dtype == torch.float8_e4m3fn
            and k.shape[0] == cache.shape[0] * 16
            and v.shape[0] == k.shape[0] - 8
            and v.data_ptr() - k.data_ptr() == 8 * 16 * 128
            and attn._aiter_sparse_pa_block_page_stride == 16,
            f"{label}: page-16 K/V views",
        )
        index = attn.indexer.index_cache.kv_cache
        check(index, "index cache", (index.shape[0], 128, 128), torch.float8_e4m3fn)
        for name, scale in (("K", attn._k_scale), ("V", attn._v_scale)):
            require(
                scale.numel() == 1
                and scale.dtype == torch.float32
                and scale.device == device
                and scale.is_contiguous()
                and bool(torch.isfinite(scale).all())
                and bool((scale > 0).all()),
                f"{label}: {name} cache scale must be a positive finite FP32 scalar",
            )
        return cls(
            layer.layer_id,
            attn,
            check(layer.input_layernorm.weight, "input norm", (6144,)),
            w_qkv,
            s_qkv,
            check(attn.q_norm.weight, "Q norm", (128,)),
            check(attn.k_norm.weight, "K norm", (128,)),
            check(attn.index_q_norm.weight, "index Q norm", (128,)),
            check(attn.index_k_norm.weight, "index K norm", (128,)),
            cos_sin,
            w_o,
            s_o,
            check(layer.post_attention_layernorm.weight, "post norm", (6144,)),
            gate,
            check(moe.e_score_correction_bias, "router bias", (128,), torch.float32),
            w13,
            s13,
            w2,
            s2,
            k,
            v,
            index,
            attn._k_scale,
            attn._v_scale,
        )

    @property
    def extra_weight_bytes(self) -> int:
        return sum(
            t.numel() * t.element_size()
            for t in (self.w_qkv, self.s_qkv, self.w_o, self.s_o, self.gate)
        )

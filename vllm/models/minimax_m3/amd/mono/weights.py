# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checked zero-copy views of MiniMax-M3 post-load tensors."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import Mxfp4MoeBackend

HIDDEN = 6144
QKV_ROWS = 2560
O_K = 2048
EXPERTS = 129
INTER = 768


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"--monokernel {message}.")


def _tensor(
    value: torch.Tensor,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    _require(
        value.dtype == dtype and tuple(value.shape) == shape and value.stride(-1) == 1,
        f"{name} is {tuple(value.shape)} {value.dtype}, expected {shape} {dtype}",
    )
    return value.data


@dataclass(frozen=True)
class MonoLayerWeights:
    input_norm: torch.Tensor
    qkv: torch.Tensor
    q_norm: torch.Tensor
    k_norm: torch.Tensor
    index_q_norm: torch.Tensor
    index_k_norm: torch.Tensor
    output: torch.Tensor
    post_norm: torch.Tensor
    router: torch.Tensor
    router_bias: torch.Tensor
    w13: torch.Tensor
    s13: torch.Tensor
    w2: torch.Tensor
    s2: torch.Tensor
    routed_scale: float
    shared_weight: float

    @classmethod
    def from_layer(cls, layer) -> MonoLayerWeights:
        layer_id = layer.layer_id
        attn = layer.self_attn
        moe = layer.block_sparse_moe
        runner = moe.experts
        experts = getattr(runner, "routed_experts", runner)
        backend = getattr(getattr(experts, "quant_method", None), "mxfp4_backend", None)

        _require(
            moe.is_fused_shared_expert_enabled and moe.use_aiter_moe_fse,
            f"layer {layer_id} requires the AITER fused shared expert",
        )
        _require(
            getattr(experts, "top_k", None) == 4,
            f"layer {layer_id} requires top-4 routing",
        )
        _require(
            backend == Mxfp4MoeBackend.AITER_MXFP4_MXFP4,
            f"layer {layer_id} requires AITER_MXFP4_MXFP4, found {backend}",
        )
        _require(
            float(moe.routed_scaling_factor) == 2.0,
            f"layer {layer_id} routed scale must be 2.0",
        )

        w13 = experts.w13_weight
        w2 = experts.w2_weight
        s13 = experts.w13_weight_scale
        s2 = experts.w2_weight_scale
        _require(
            w13.dtype == torch.float4_e2m1fn_x2 and w2.dtype == torch.float4_e2m1fn_x2,
            f"layer {layer_id} experts are not MXFP4",
        )
        _require(
            getattr(w13, "is_shuffled", False) and getattr(w2, "is_shuffled", False),
            f"layer {layer_id} expert weights are not AITER-shuffled",
        )
        _require(
            w13.numel() * w13.element_size() == EXPERTS * 2 * INTER * HIDDEN // 2,
            f"layer {layer_id} w13 byte count is unsupported",
        )
        _require(
            w2.numel() * w2.element_size() == EXPERTS * HIDDEN * INTER // 2,
            f"layer {layer_id} w2 byte count is unsupported",
        )
        _require(
            s13.dtype == torch.uint8
            and tuple(s13.shape) == (EXPERTS * 2 * INTER, HIDDEN // 32)
            and s13.stride(-1) == 1,
            f"layer {layer_id} w13 scales are not shuffled E8M0 group-32",
        )
        _require(
            s2.dtype == torch.uint8
            and tuple(s2.shape) == (EXPERTS * HIDDEN, INTER // 32)
            and s2.stride(-1) == 1,
            f"layer {layer_id} w2 scales are not shuffled E8M0 group-32",
        )

        vec = lambda value, size, name: _tensor(  # noqa: E731
            value, shape=(size,), dtype=torch.bfloat16, name=name
        )
        return cls(
            input_norm=vec(
                layer.input_layernorm.weight, HIDDEN, f"layer {layer_id} input norm"
            ),
            qkv=_tensor(
                attn.qkv_proj.weight,
                shape=(QKV_ROWS, HIDDEN),
                dtype=torch.bfloat16,
                name=f"layer {layer_id} QKV/index projection",
            ),
            q_norm=vec(attn.q_norm.weight, 128, f"layer {layer_id} Q norm"),
            k_norm=vec(attn.k_norm.weight, 128, f"layer {layer_id} K norm"),
            index_q_norm=vec(
                attn.index_q_norm.weight, 128, f"layer {layer_id} index Q norm"
            ),
            index_k_norm=vec(
                attn.index_k_norm.weight, 128, f"layer {layer_id} index K norm"
            ),
            output=_tensor(
                attn.o_proj.weight,
                shape=(HIDDEN, O_K),
                dtype=torch.bfloat16,
                name=f"layer {layer_id} output projection",
            ),
            post_norm=vec(
                layer.post_attention_layernorm.weight,
                HIDDEN,
                f"layer {layer_id} post-attention norm",
            ),
            router=_tensor(
                moe.gate.weight,
                shape=(128, HIDDEN),
                dtype=torch.float32,
                name=f"layer {layer_id} router",
            ),
            router_bias=_tensor(
                moe.e_score_correction_bias,
                shape=(128,),
                dtype=torch.float32,
                name=f"layer {layer_id} router bias",
            ),
            w13=w13.data,
            s13=s13.data,
            w2=w2.data,
            s2=s2.data,
            routed_scale=2.0,
            shared_weight=1.0,
        )

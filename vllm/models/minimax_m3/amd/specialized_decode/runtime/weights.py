# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checked zero-copy views of the tensors read by the fused MiniMax-M3 FFN."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch

from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe import (
    AiterMxfp4Experts,
)
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import Mxfp4MoeBackend

HIDDEN = 6144
EXPERTS = 129
INTER = 768


class EffectiveActivationMode(Enum):
    BF16 = "bf16"
    MXFP4 = "mxfp4"
    UNSUPPORTED = "unsupported"


def effective_activation_mode(experts) -> EffectiveActivationMode:
    quant_method = getattr(experts, "quant_method", None)
    moe_kernel = getattr(quant_method, "moe_kernel", None)
    fused_experts = getattr(moe_kernel, "fused_experts", None)
    if not isinstance(fused_experts, AiterMxfp4Experts):
        return EffectiveActivationMode.UNSUPPORTED
    if not fused_experts.expects_unquantized_inputs:
        return EffectiveActivationMode.MXFP4
    if getattr(fused_experts.moe_config, "use_mxfp4_w4a4_dsv4", False):
        return EffectiveActivationMode.MXFP4
    return EffectiveActivationMode.BF16


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"--monokernel {message}.")


def _tensor(
    value: torch.Tensor,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    name: str,
    require_cuda: bool,
) -> torch.Tensor:
    expected_stride = torch.empty(shape, device="meta").stride()
    _require(
        value.dtype == dtype
        and tuple(value.shape) == shape
        and value.stride() == expected_stride
        and value.storage_offset() == 0
        and (not require_cuda or value.device.type == "cuda"),
        f"{name} has unsupported shape/dtype/stride/storage/device",
    )
    return value.detach()


def _bf16_vector(
    value: torch.Tensor, size: int, name: str, require_cuda: bool
) -> torch.Tensor:
    return _tensor(
        value,
        shape=(size,),
        dtype=torch.bfloat16,
        name=name,
        require_cuda=require_cuda,
    )


@dataclass(frozen=True)
class MonoLayerWeights:
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
    def from_layer(cls, layer, *, require_cuda: bool = True) -> MonoLayerWeights:
        layer_id = layer.layer_id
        moe = layer.block_sparse_moe
        runner = moe.experts
        experts = getattr(runner, "routed_experts", runner)

        _require(
            moe.is_fused_shared_expert_enabled and moe.use_aiter_moe_fse,
            f"layer {layer_id} requires the AITER fused shared expert",
        )
        _require(
            getattr(experts, "top_k", None) == 4,
            f"layer {layer_id} requires top-4 routing",
        )
        # the kernel reads e8m0_shuffle scales and (16, 16) shuffle_weight
        # weights, which only this backend's post-load conversion produces
        _require(
            getattr(experts.quant_method, "mxfp4_backend", None)
            is Mxfp4MoeBackend.AITER_MXFP4_MXFP4,
            f"layer {layer_id} requires the AITER_MXFP4_MXFP4 expert layout",
        )
        _require(
            effective_activation_mode(experts) is EffectiveActivationMode.BF16,
            f"layer {layer_id} requires effective BF16 expert activations",
        )
        _require(
            float(moe.routed_scaling_factor) == 2.0,
            f"layer {layer_id} routed scale must be 2.0",
        )
        _require(
            getattr(experts, "renormalize", None) is True,
            f"layer {layer_id} routed weights must be renormalized",
        )
        _require(
            getattr(experts, "scoring_func", None) == "sigmoid",
            f"layer {layer_id} router must use sigmoid scores",
        )
        _require(
            float(getattr(experts, "swiglu_alpha", 0.0)) == 1.702
            and float(getattr(experts, "swiglu_beta", 0.0)) == 1.0
            and float(getattr(experts, "swiglu_limit", 0.0)) == 7.0,
            f"layer {layer_id} requires SwiGLU-OAI alpha=1.702 beta=1 clamp=7",
        )
        _require(
            getattr(moe, "n_shared_experts", None) == 1,
            f"layer {layer_id} requires one shared expert at slot 128",
        )
        _require(
            type(layer.post_attention_layernorm).__name__ == "MiniMAXGemmaRMSNorm",
            f"layer {layer_id} requires Gemma normalization",
        )
        _require(
            layer.post_attention_layernorm.variance_epsilon == 1e-6,
            f"layer {layer_id} requires Gemma norm epsilon 1e-6",
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
            tuple(w13.shape) == (EXPERTS, 2 * INTER, HIDDEN // 2)
            and w13.is_contiguous()
            and w13.storage_offset() == 0,
            f"layer {layer_id} w13 layout is unsupported",
        )
        _require(
            tuple(w2.shape) == (EXPERTS, HIDDEN, INTER // 2)
            and w2.is_contiguous()
            and w2.storage_offset() == 0,
            f"layer {layer_id} w2 layout is unsupported",
        )
        _require(
            s13.dtype == torch.uint8
            and tuple(s13.shape) == (EXPERTS, 2 * INTER, HIDDEN // 32)
            and s13.is_contiguous()
            and s13.storage_offset() == 0,
            f"layer {layer_id} w13 scales are not shuffled E8M0 group-32",
        )
        _require(
            s2.dtype == torch.uint8
            and tuple(s2.shape) == (EXPERTS, HIDDEN, INTER // 32)
            and s2.is_contiguous()
            and s2.storage_offset() == 0,
            f"layer {layer_id} w2 scales are not shuffled E8M0 group-32",
        )
        _require(
            not require_cuda
            or all(t.device.type == "cuda" for t in (w13, w2, s13, s2)),
            f"layer {layer_id} expert tensors must be on GPU",
        )
        return cls(
            post_norm=_bf16_vector(
                layer.post_attention_layernorm.weight,
                HIDDEN,
                f"layer {layer_id} post-attention norm",
                require_cuda,
            ),
            router=_tensor(
                moe.gate.weight,
                shape=(128, HIDDEN),
                dtype=torch.float32,
                name=f"layer {layer_id} router",
                require_cuda=require_cuda,
            ),
            router_bias=_tensor(
                moe.e_score_correction_bias,
                shape=(128,),
                dtype=torch.float32,
                name=f"layer {layer_id} router bias",
                require_cuda=require_cuda,
            ),
            w13=w13.detach(),
            s13=s13.detach(),
            w2=w2.detach(),
            s2=s2.detach(),
            routed_scale=2.0,
            shared_weight=1.0,
        )

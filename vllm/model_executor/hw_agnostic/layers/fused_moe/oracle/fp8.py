# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP8 MoE backend selector — Triton-only.

The upstream selector dispatches across DeepGEMM / FlashInfer / Marlin /
AITER / CUTLASS / XPU / CPU. The hw-agnostic path supports Triton
(plus the BatchedTriton variant when the activation format is
batched). Vendor backends are rejected by ``MoERunner._validate_supported_settings``;
this file is the kernel-class lookup the FP8 method consumes.
"""
from enum import Enum

import torch

import vllm.model_executor.hw_agnostic.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.hw_agnostic.layers.fused_moe.all2all_utils import (
    maybe_make_prepare_finalize,
)
from vllm.model_executor.hw_agnostic.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
    fp8_w8a8_moe_quant_config,
)

logger = init_logger(__name__)


class Fp8MoeBackend(Enum):
    TRITON = "TRITON"
    BATCHED_TRITON = "BATCHED_TRITON"


def select_fp8_moe_backend(
    config: FusedMoEConfig,
    weight_key=None,
    activation_key=None,
) -> tuple[Fp8MoeBackend, type[mk.FusedMoEExperts]]:
    """Return (backend, experts_cls). Always Triton on the hw-agnostic path."""
    from vllm.model_executor.hw_agnostic.layers.fused_moe.experts.triton_moe import (
        TritonExperts,
    )

    return Fp8MoeBackend.TRITON, TritonExperts


def convert_to_fp8_moe_kernel_format(
    fp8_backend: Fp8MoeBackend,
    layer,
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    w13_input_scale: torch.Tensor | None,
    w2_input_scale: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """No-op weight reformat for the Triton path (kernel consumes the
    standard ``[E, N, K]`` / ``[E, K, N]`` layouts directly)."""
    return w13, w2, w13_scale, w2_scale


def make_fp8_moe_quant_config(
    fp8_backend: Fp8MoeBackend,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    a1_scale: torch.Tensor | None,
    a2_scale: torch.Tensor | None,
    w1_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
    block_shape: list[int] | None = None,
    per_act_token_quant: bool = False,
    per_out_ch_quant: bool = False,
    swiglu_limit: float | None = None,
    gemm1_alpha: float | None = None,
    gemm1_beta: float | None = None,
) -> FusedMoEQuantConfig:
    return fp8_w8a8_moe_quant_config(
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        w1_bias=w1_bias,
        w2_bias=w2_bias,
        per_act_token_quant=per_act_token_quant,
        per_out_ch_quant=per_out_ch_quant,
        block_shape=block_shape,
        gemm1_clamp_limit=swiglu_limit,
    )


def make_fp8_moe_kernel(
    moe_quant_config: FusedMoEQuantConfig,
    moe_config: FusedMoEConfig,
    experts_cls: type[mk.FusedMoEExperts],
    fp8_backend: Fp8MoeBackend,
    routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> mk.FusedMoEKernel:
    prepare_finalize = maybe_make_prepare_finalize(moe_config)
    logger.info_once("Using %s", prepare_finalize.__class__.__name__)
    experts = experts_cls(moe_config=moe_config, quant_config=moe_quant_config)
    return mk.FusedMoEKernel(prepare_finalize, experts)

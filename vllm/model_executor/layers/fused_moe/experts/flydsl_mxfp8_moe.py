# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP8 (1x32 block, E8M0) MoE via AITER's FlyDSL two-stage grouped GEMM
(gfx950); alternative to ``Mxfp8NativeTritonExperts``. Routes through
``aiter.fused_moe`` (per_1x32, gate_mode=INTERLEAVE); weights are preshuffled in
``convert_to_fp8_moe_kernel_format``.
"""

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.experts.mxfp8_emulation_moe import (
    Mxfp8TritonExpertsBase,
)
from vllm.platforms import current_platform

logger = init_logger(__name__)


def is_flydsl_mxfp8_moe_available() -> bool:
    """True when the FlyDSL MXFP8 MoE can run here (gfx950 + ``flydsl``
    importable). This is capability only; the oracle adds the aiter-MoE-switch
    gate for auto-selection (``--moe-backend flydsl`` selects it directly)."""
    if not (current_platform.is_rocm() and current_platform.supports_mx()):
        return False
    try:
        from aiter.ops.flydsl.utils import is_flydsl_available

        return bool(is_flydsl_available())
    except Exception:
        return False


class FlydslMxfp8Experts(Mxfp8TritonExpertsBase):
    """MXFP8 MoE through AITER's FlyDSL two-stage grouped GEMM (gfx950)."""

    @property
    def quant_dtype(self) -> torch.dtype | str | None:
        return self.quant_config.quant_dtype

    @property
    def block_shape(self) -> list[int] | None:
        return self.quant_config.block_shape

    @property
    def expects_unquantized_inputs(self) -> bool:
        # aiter.fused_moe MXFP8-quantizes the activations internally.
        return True

    @staticmethod
    def _supports_current_device() -> bool:
        return is_flydsl_mxfp8_moe_available()

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        from aiter import ActivationType, QuantType, dtypes
        from aiter.fused_moe import fused_moe
        from aiter.ops.flydsl.moe_common import GateMode

        if expert_map is not None:
            raise NotImplementedError(
                "FlydslMxfp8Experts does not support expert parallelism yet; "
                "disable EP or use the native MXFP8 MoE backend."
            )

        # Re-tag the preshuffled weights: replace_parameter drops the
        # is_shuffled flag, without which aiter picks a broken CK kernel.
        w1.is_shuffled = True
        w2.is_shuffled = True

        limit = self.quant_config.gemm1_clamp_limit
        swiglu_limit = 0.0 if limit is None else float(limit)

        assert self.w1_scale_val is not None and self.w2_scale_val is not None
        w1_scale = self.w1_scale_val.view(dtypes.fp8_e8m0)
        w2_scale = self.w2_scale_val.view(dtypes.fp8_e8m0)

        # aiter requires FP32 routing weights / INT32 ids.
        out = fused_moe(
            hidden_states,
            w1,
            w2,
            topk_weights.to(torch.float32),
            topk_ids.to(torch.int32),
            expert_mask=None,
            activation=ActivationType.Swiglu,
            quant_type=QuantType.per_1x32,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            a1_scale=None,
            a2_scale=None,
            doweight_stage1=apply_router_weight_on_input,
            swiglu_limit=swiglu_limit,
            gate_mode=GateMode.INTERLEAVE.value,
        )
        output.copy_(out.to(output.dtype))

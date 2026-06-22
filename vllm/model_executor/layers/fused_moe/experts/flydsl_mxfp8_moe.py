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
        # Device capability only (gfx950 / MX-capable ROCm). The flydsl package
        # check lives in is_supported_config so a missing package is reported
        # distinctly from an unsupported device.
        return current_platform.is_rocm() and current_platform.supports_mx()

    @staticmethod
    def _supports_parallel_config(moe_parallel_config) -> bool:
        # Both TP (expert_map=None) and EP are supported: apply() forwards the
        # expert_map as aiter's ``expert_mask`` (the per-rank local-expert
        # selection), mirroring the native rocm_aiter_moe path.
        return True

    @staticmethod
    def is_supported_config(
        cls, moe_config, weight_key, activation_key, activation_format
    ):
        is_supported, reason = super().is_supported_config(
            cls, moe_config, weight_key, activation_key, activation_format
        )
        # _supports_current_device() only gates on the device; surface a clear
        # reason when the device is fine but the flydsl package is missing.
        if is_supported and not is_flydsl_mxfp8_moe_available():
            return False, (
                "kernel requires the aiter flydsl package, which is not installed"
            )
        return is_supported, reason

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
        from aiter.ops.flydsl.moe_common import GateMode

        from vllm._aiter_ops import rocm_aiter_ops

        # Re-tag the preshuffled weights: replace_parameter drops the
        # is_shuffled flag, without which aiter picks a broken CK kernel.
        w1.is_shuffled = True
        w2.is_shuffled = True

        limit = self.quant_config.gemm1_clamp_limit
        swiglu_limit = 0.0 if limit is None else float(limit)

        assert self.w1_scale_val is not None and self.w2_scale_val is not None
        w1_scale = self.w1_scale_val.view(dtypes.fp8_e8m0)
        w2_scale = self.w2_scale_val.view(dtypes.fp8_e8m0)

        # Under EP, aiter expects ``expert_mask`` as a 0/1 *local-expert* mask
        # over global ids with a trailing fake-expert sentinel slot
        # (shape ``[global_num_experts + 1]``), NOT vLLM's expert_map (a
        # global->local index map with -1 for non-local). Convert it; aiter
        # derives the global->local compaction from the mask itself. ``None``
        # under pure TP.
        if expert_map is not None:
            local_mask = (expert_map >= 0).to(torch.int32)
            expert_mask = torch.cat([local_mask, local_mask.new_zeros(1)])
        else:
            expert_mask = None

        # Route through the graph-safe ``rocm_aiter_fused_moe`` custom op so the
        # call is captured under HIP graphs / torch.compile (a direct
        # ``aiter.fused_moe`` is opaque to the dispatcher). aiter requires FP32
        # routing weights / INT32 ids.
        out = rocm_aiter_ops.fused_moe(
            hidden_states,
            w1,
            w2,
            topk_weights.to(torch.float32),
            topk_ids.to(torch.int32),
            expert_mask=expert_mask,
            activation_method=ActivationType.Swiglu.value,
            quant_method=QuantType.per_1x32.value,
            doweight_stage1=apply_router_weight_on_input,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            a1_scale=None,
            a2_scale=None,
            gate_mode=GateMode.INTERLEAVE.value,
            swiglu_limit=swiglu_limit,
            output_dtype=output.dtype,
        )
        output.copy_(out.to(output.dtype))

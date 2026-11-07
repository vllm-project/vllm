# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit CRI MXFP activation controls for packed MXFP4 checkpoints."""

from typing import Any

import torch

from vllm import envs
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import FusedMoeWeightScaleSupported
from vllm.model_executor.layers.fused_moe.activation import (
    ApplyMoEActivationConfig,
    MoEActivation,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.utils import (
    move_parameter,
    replace_parameter,
    set_weight_attrs,
)

from .inc_wna16_moe import INCARKWNA16MoEMethod

logger = init_logger(__name__)


def dequantize_mxfp4(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Decode checkpoint low-nibble-first E2M1 and E8M0 independently."""
    levels = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float32,
        device=codes.device,
    )
    raw = codes.to(torch.int32)
    raw = torch.stack((raw & 15, raw >> 4), dim=-1).flatten(-2)
    values = levels[raw & 7] * torch.where((raw & 8) != 0, -1.0, 1.0)
    exponents = (scales.to(torch.int32) - 127).repeat_interleave(32, dim=-1)
    return torch.ldexp(values, exponents)


class INCARKMxfp4MoEMethod(INCARKWNA16MoEMethod):
    """Reuse ARK's XPU routing/activation/gather with native MXFP GEMMs.

    Only eager TP=1, EP=1, unshared SiLU experts are supported. Both expert
    projections use the same selected activation format in prefill and decode.
    """

    def __init__(self, moe: FusedMoEConfig, backend: str) -> None:
        FusedMoEMethodBase.__init__(self, moe)
        if backend not in ("ark_mxfp8", "ark_mxfp4", "ark_mxfp4_fwht"):
            raise ValueError(f"Unsupported ARK MXFP4 backend: {backend}")
        if (
            moe.tp_size != 1
            or moe.ep_size != 1
            or moe.dp_size != 1
            or moe.sp_size != 1
            or moe.shared_expert_prefix is not None
            or moe.activation != MoEActivation.SILU
            or moe.has_bias
        ):
            raise NotImplementedError(
                "ARK MXFP4 MoE supports unshared, bias-free SiLU, TP/EP/DP/SP=1"
            )
        import auto_round_kernel as ark

        lib = getattr(ark, "xpu_lib", None)
        if getattr(lib, "sycl_target", None) not in ("cri", "intel_gpu_cri"):
            raise NotImplementedError("ARK MXFP4 MoE requires a CRI ARK build")
        required = (
            "moe_gemm_prefill_mxfp4_mxfp4",
            "moe_gemm_prefill_mxfp8_mxfp4",
            "moe_gemm_prefill_hmt_mxfp4_mxfp4",
            "mxfp8_quant",
        )
        if not all(hasattr(ark, name) for name in required):
            raise ImportError("ARK native MXFP quantization/GEMM APIs are missing")
        self.use_prepacked_weights = (
            envs.VLLM_XPU_INC_MXFP8_PREPACK
            if backend == "ark_mxfp8"
            else envs.VLLM_XPU_INC_MXFP4_PREPACK
        )
        prepared_prefix = "mxfp8_mxfp4" if backend == "ark_mxfp8" else "mxfp4_mxfp4"
        if self.use_prepacked_weights and not all(
            hasattr(ark, name)
            for name in (
                f"moe_{prepared_prefix}_prepack",
                f"moe_gemm_prefill_{prepared_prefix}_prepacked",
            )
        ):
            raise ImportError("ARK prepared MXFP weight APIs are missing")
        self.ark = ark
        self.backend = backend
        self.prepared_weights: dict[str, Any] = {}
        self.group_size = 32
        self.local_num_experts = None
        self.global_num_experts = None
        self.expert_map = None
        self.rows_per_expert = None
        self.unpermuted_row_to_permuted_row = None
        self.router_weight_ones = None
        self.router_weights_fn = self._keep_router_weights
        self.activation = moe.activation
        self.activation_config = ApplyMoEActivationConfig.from_configs(
            moe, FusedMoEQuantConfig.make()
        )
        self.inter_size = moe.intermediate_size_per_partition
        self.inter_size_scale = 1
        self.remap_hidden_states_op = torch.ops._moe_C.remap_hidden_states
        self.moe_gather_op = torch.ops._moe_C.moe_gather
        self.w13_codes = None
        self.w13_scales = None
        self.w2_codes = None
        self.w2_scales = None
        logger.info_once("Using native CRI ARK MoE activation control: %s", backend)

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        layer.num_experts = num_experts
        layer.params_dtype = params_dtype
        projections = (
            (
                "w13",
                self.moe.w13_num_shards * intermediate_size_per_partition,
                hidden_size,
            ),
            ("w2", hidden_size, intermediate_size_per_partition),
        )
        for projection, n, k in projections:
            for suffix, width in (("packed", k // 2), ("scale", k // 32)):
                parameter = torch.nn.Parameter(
                    torch.empty(num_experts, n, width, dtype=torch.uint8),
                    requires_grad=False,
                )
                layer.register_parameter(f"{projection}_weight_{suffix}", parameter)
                attrs = dict(extra_weight_attrs)
                if suffix == "scale":
                    attrs["quant_method"] = FusedMoeWeightScaleSupported.GROUP.value
                set_weight_attrs(parameter, attrs)

    def get_fused_moe_quant_config(self, layer) -> None:
        # Quantization is performed inside the ARK calls, not the modular kernel.
        return None

    def process_weights_after_loading(self, layer) -> None:
        for projection in ("w13", "w2"):
            move_parameter(layer, f"{projection}_weight_packed", f"{projection}_weight")
            weight = getattr(layer, f"{projection}_weight").detach().contiguous()
            scales = getattr(layer, f"{projection}_weight_scale").detach().contiguous()
            if weight.dtype != torch.uint8 or scales.dtype != torch.uint8:
                raise ValueError("ARK MXFP4 requires uint8 E2M1/E8M0 weights")
            if weight.shape[-1] * 2 % 64 or scales.shape[-1] * 16 != weight.shape[-1]:
                raise ValueError("ARK MXFP4 requires K divisible by 64 and group32")
            if self.backend == "ark_mxfp4_fwht":
                from auto_round_kernel.mxfp4_hadamard import mxfp4_hadamard_quant

                # Bound temporary storage: W@H32 followed by group32 requantization.
                for expert in range(weight.shape[0]):
                    source = dequantize_mxfp4(weight[expert], scales[expert])
                    rotated, new_scales = mxfp4_hadamard_quant(
                        source.to(torch.bfloat16)
                    )
                    weight[expert].copy_(rotated)
                    scales[expert].copy_(new_scales)
            replace_parameter(layer, f"{projection}_weight", weight)
            replace_parameter(layer, f"{projection}_weight_scale", scales)
            setattr(self, f"{projection}_codes", weight)
            setattr(self, f"{projection}_scales", scales)
            if self.use_prepacked_weights:
                prepare = (
                    self.ark.moe_mxfp8_mxfp4_prepack
                    if self.backend == "ark_mxfp8"
                    else self.ark.moe_mxfp4_mxfp4_prepack
                )
                self.prepared_weights[projection] = prepare(
                    weight, scales, group_size=32
                )

        if layer.expert_map is not None or layer.apply_router_weight_on_input:
            raise NotImplementedError(
                "ARK MXFP4 requires local output-weighted routing"
            )
        assert self.w13_codes is not None
        self.local_num_experts = self.w13_codes.shape[0]
        self.global_num_experts = layer.global_num_experts
        self.expert_map = layer.expert_map
        self.rows_per_expert = torch.empty(
            self.local_num_experts, dtype=torch.int32, device=self.w13_codes.device
        )

    def _check_moe_weights_loaded(self) -> None:
        assert self.w13_codes is not None and self.w2_codes is not None
        assert self.w13_scales is not None and self.w2_scales is not None

    def _project(self, x, codes, scales, routing):
        kwargs = {"output_dtype": x.dtype, "group_size": 32}
        if self.backend == "ark_mxfp4_fwht":
            return self.ark.moe_gemm_prefill_hmt_mxfp4_mxfp4(
                x, codes, scales, routing, **kwargs
            )
        if self.backend == "ark_mxfp8":
            quantized, act_scales = self.ark.mxfp8_quant(x)
            gemm = self.ark.moe_gemm_prefill_mxfp8_mxfp4
        else:
            from auto_round_kernel.mxfp4_hadamard import mxfp4_hadamard_quant

            quantized, act_scales = mxfp4_hadamard_quant(x, _quant_only=True)
            gemm = self.ark.moe_gemm_prefill_mxfp4_mxfp4
        return gemm(quantized, act_scales, codes, scales, routing, **kwargs)

    def _apply_w13_moe(self, x, rows_per_expert):
        if self.use_prepacked_weights:
            return self._apply_prepared_projection(x, "w13", rows_per_expert)
        return self._project(x, self.w13_codes, self.w13_scales, rows_per_expert)

    def _apply_w2_moe(self, x, rows_per_expert):
        if self.use_prepacked_weights:
            return self._apply_prepared_projection(x, "w2", rows_per_expert)
        return self._project(x, self.w2_codes, self.w2_scales, rows_per_expert)

    def _apply_prepared_projection(self, x, projection, routing):
        prepared = self.prepared_weights[projection]
        if self.backend == "ark_mxfp8":
            return self.ark.moe_gemm_prefill_mxfp8_mxfp4_prepacked(
                *self.ark.mxfp8_quant(x), prepared, routing
            )
        if self.backend == "ark_mxfp4_fwht":
            return self.ark.moe_gemm_prefill_hmt_mxfp4_mxfp4_prepacked(
                x, prepared, routing
            )
        from auto_round_kernel.mxfp4_hadamard import mxfp4_hadamard_quant

        return self.ark.moe_gemm_prefill_mxfp4_mxfp4_prepacked(
            *mxfp4_hadamard_quant(x, _quant_only=True), prepared, routing
        )

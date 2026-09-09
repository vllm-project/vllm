# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
OCP MX quantization emulation for MoE.

This file implements OCP MX (MXFP4/MXFP6) emulation for MoE in case the
hardware used does not natively support OCP MX MoE.

Weights are dequantized on the fly during each forward, we fall back to calling
`TritonExperts` using BF16, and fake OCP MX quantize-dequantize
is applied on activations via `moe_kernel_quantize_input`.
"""

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
from vllm.model_executor.layers.quantization.utils.mxfp4_utils import dequant_mxfp4
from vllm.model_executor.layers.quantization.utils.mxfp6_utils import dequant_mxfp6
from vllm.model_executor.layers.quantization.utils.ocp_mx_utils import (
    _ACTIVATION_QUANT_DTYPE_MAP,
    _ACTIVATION_QUANT_KEY_MAP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8StaticTensorSym,
)
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_quark

logger = init_logger(__name__)


class OCP_MXQuantizationEmulationTritonExperts(TritonExperts):
    """
    Extension of TritonExperts to support emulated OCP MX MoE experts.

    It may be used for OCP MX (MXFP4/MXFP6) models when the device does not
    have native support for these dtypes.
    """

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(moe_config, quant_config)
        logger.warning_once(
            "Using OCP_MXQuantizationEmulationTritonExperts MOE backend. This"
            " will dequantize weights on the fly and may be slower than native"
            " quantized MOE. Consider using a device with native OCP MX"
            " quantization support for better performance."
        )

        assert self.weight_quant_dtype == quant_config._w2.dtype

        # `TritonExperts.apply` expects pre-dequantized weights,
        # which we handle in `apply` below.
        self.w1_scale_val = self.quant_config.w1_scale
        self.w2_scale_val = self.quant_config.w2_scale

        self.quant_config._w1.scale = None
        self.quant_config._w2.scale = None

        self.quantization_emulation = True

        activation_dtype = quant_config._a1.dtype
        assert activation_dtype is None or isinstance(activation_dtype, str)

        self.activation_quant_key = (
            None
            if activation_dtype is None
            else _ACTIVATION_QUANT_DTYPE_MAP[activation_dtype]
        )

        # TODO: Migrate quant_config._a1.dtype to be QuantKey directly.
        if self.activation_quant_key is None:
            self._quant_dtype = None
        elif self.activation_quant_key == kFp8StaticTensorSym:
            self._quant_dtype = current_platform.fp8_dtype()
        else:
            self._quant_dtype = _ACTIVATION_QUANT_KEY_MAP[self.activation_quant_key]

    @staticmethod
    def is_supported_config(
        cls: type["mk.FusedMoEExperts"],
        moe_config: FusedMoEConfig,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
        activation_format: mk.FusedMoEActivationFormat,
    ) -> tuple[bool, str | None]:
        if not has_quark():
            return False, "kernel requires amd-quark package"

        return TritonExperts.is_supported_config(
            cls=cls,
            moe_config=moe_config,
            weight_key=weight_key,
            activation_key=activation_key,
            activation_format=activation_format,
        )

    @property
    def quant_dtype(self) -> torch.dtype | str | None:
        return self._quant_dtype

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

    @staticmethod
    def _supports_quant_scheme(
        weight_key,
        activation_key,
    ) -> bool:
        # This class is used for emulation only - the oracle selects it
        # directly rather than via quant scheme matching.
        return True

    def _dequantize_weights(
        self,
        w: torch.Tensor,
        w_scale: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Dequantize weights according to their quantization descriptor."""
        if self.weight_quant_dtype == "mxfp4":
            return dequant_mxfp4(w, w_scale, dtype)
        if self.weight_quant_dtype == "mxfp6_e3m2":
            return dequant_mxfp6(w, w_scale, quant_dtype="fp6_e3m2", float_dtype=dtype)
        if self.weight_quant_dtype == "mxfp6_e2m3":
            return dequant_mxfp6(w, w_scale, quant_dtype="fp6_e2m3", float_dtype=dtype)
        raise NotImplementedError(
            f"Unsupported OCP MX weight dtype {self.weight_quant_dtype}"
        )

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        """
        Apply emulated quantized MoE computation.

        This dequantizes the weights on the fly and calls TritonExperts.apply
        with activation quantization support.
        """
        assert w1.dtype == torch.uint8
        assert w2.dtype == torch.uint8

        # Dequantize w1 and w2 from packed OCP MX format to bf16/fp16
        w1_dequant = self._dequantize_weights(
            w1, self.w1_scale_val, hidden_states.dtype
        )
        w2_dequant = self._dequantize_weights(
            w2, self.w2_scale_val, hidden_states.dtype
        )

        # Activation quantization/dequantization is deferred to
        # `moe_kernel_quantize_input` in TritonExperts.apply.
        super().apply(
            output=output,
            hidden_states=hidden_states,
            w1=w1_dequant,
            w2=w2_dequant,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=activation,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            a1q_scale=None,
            a2_scale=None,
            workspace13=workspace13,
            workspace2=workspace2,
            expert_tokens_meta=expert_tokens_meta,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )

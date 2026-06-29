# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch

from vllm.config import get_current_vllm_config
from vllm.model_executor.hw_agnostic.layers.attention import Attention
from vllm.model_executor.hw_agnostic.layers.fused_moe.routed_experts import (
    RoutedExperts,
)
from vllm.model_executor.hw_agnostic.layers.fused_moe.unquantized_fused_moe_method import (  # noqa: E501
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.hw_agnostic.layers.linear import (
    LinearBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.hw_agnostic.quantization.fp8_config import Fp8Config
from vllm.model_executor.hw_agnostic.quantization.fp8_kv_cache import (
    Fp8KVCacheMethod,
)
from vllm.model_executor.hw_agnostic.quantization.fp8_linear_method import (
    Fp8LinearMethod,
)
from vllm.model_executor.hw_agnostic.quantization.fp8_moe_method import (
    Fp8MoEMethod,
    Fp8OnlineMoEMethod,
)
from vllm.model_executor.hw_agnostic.quantization.online_fp8_linear_method import (  # noqa: E501
    Fp8PerTensorOnlineLinearMethod,
)
from vllm.model_executor.hw_agnostic.quantization.quant_keys import (
    is_layer_skipped,
)

# Only fp8 is supported on the hw-agnostic DSv4 path; FP4 (NVFP4 / MXFP4)
# needs hw-specific kernels (Cutlass / FlashInfer / TRT-LLM / AITER) with no
# Triton equivalent today.
_DEEPSEEK_V4_EXPERT_DTYPES = ("fp8",)

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.base_config import (
        QuantizeMethodBase,
    )


class DeepseekV4FP8Config(Fp8Config):
    """DSv4 FP8 config (hw-agnostic; expert_dtype must be fp8)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._resolved_expert_dtype: str | None = None

    @property
    def expert_dtype(self) -> str:
        if self._resolved_expert_dtype is None:
            try:
                hf_config = get_current_vllm_config().model_config.hf_config
            except Exception:
                # vllm_config not yet set — retry on a later call.
                return "fp8"
            expert_dtype = getattr(hf_config, "expert_dtype", "fp8")
            if expert_dtype not in _DEEPSEEK_V4_EXPERT_DTYPES:
                raise ValueError(
                    f"Unsupported DeepSeek V4 expert_dtype={expert_dtype!r} on the "
                    "hw-agnostic path; expected 'fp8'. FP4 expert dtypes "
                    "(NVFP4 / MXFP4) require hw-specific kernels and are "
                    "only available via the hw-specific quantization path."
                )
            self._resolved_expert_dtype = expert_dtype
            from vllm.logger import init_logger

            init_logger(__name__).info_once(
                "DeepSeek V4 expert_dtype resolved to %r", expert_dtype
            )
        return self._resolved_expert_dtype

    @classmethod
    def get_name(cls) -> Literal["deepseek_v4_fp8"]:
        # Inlined literal: avoids a module-level import of vLLM's
        # ``QuantizationMethods`` typing alias.
        return "deepseek_v4_fp8"

    @classmethod
    def override_quantization_method(
        cls, hf_quant_cfg, user_quant, hf_config=None
    ) -> str | None:
        if not (
            isinstance(hf_quant_cfg, dict)
            and hf_quant_cfg.get("quant_method") in ("fp8", "deepseek_v4_fp8")
        ):
            return None
        model_type = getattr(hf_config, "model_type", None)
        if model_type == "deepseek_v4" or user_quant == "deepseek_v4_fp8":
            return "deepseek_v4_fp8"
        return None

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        if isinstance(layer, RoutedExperts):
            if is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                return UnquantizedFusedMoEMethod(layer.moe_config)
            # Triggers the FP4-rejection check eagerly.
            _ = self.expert_dtype
            if self.is_checkpoint_fp8_serialized:
                return Fp8MoEMethod(self, layer)
            return Fp8OnlineMoEMethod(self, layer)

        if isinstance(layer, LinearBase):
            if is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                return UnquantizedLinearMethod()
            if not self.is_checkpoint_fp8_serialized:
                return Fp8PerTensorOnlineLinearMethod()
            return Fp8LinearMethod(self)

        if isinstance(layer, Attention):
            return Fp8KVCacheMethod(self)

        return None

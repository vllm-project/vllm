# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.fp8 import (
    Fp8Config,
    Fp8KVCacheMethod,
    Fp8MoEMethod,
    Fp8OnlineMoEMethod,
)
from vllm.models.deepseek_v4.hw_agnostic.quantization.fp8_linear_method import (
    Fp8LinearMethod,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped,
)
from vllm.models.deepseek_v4.hw_agnostic.shared.layers.fused_moe.routed_experts import (
    RoutedExperts,
)
from vllm.models.deepseek_v4.hw_agnostic.shared.layers.fused_moe.unquantized_fused_moe_method import (  # noqa: E501
    UnquantizedFusedMoEMethod,
)
from vllm.models.deepseek_v4.hw_agnostic.shared.layers.linear import (
    LinearBase,
    UnquantizedLinearMethod,
)

_DEEPSEEK_V4_EXPERT_DTYPES = ("fp4", "fp8")

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.base_config import (
        QuantizeMethodBase,
    )
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptNvFp4Config,
    )


class DeepseekV4FP8Config(Fp8Config):
    """DSv4 FP8 config with expert-dtype-aware MoE dispatch.

    Linear / attention layers are always FP8 block quant. ``expert_dtype``
    is ``"fp4"`` (MXFP4 experts, ue8m0 FP8 linear scales) or ``"fp8"``
    (FP8 block experts, float32 FP8 linear scales).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._resolved_expert_dtype: str | None = None
        self._resolved_moe_quant_algo: str | None = None
        self._nvfp4_config: ModelOptNvFp4Config | None = None

    @property
    def expert_dtype(self) -> str:
        if self._resolved_expert_dtype is None:
            try:
                hf_config = get_current_vllm_config().model_config.hf_config
            except Exception:
                # vllm_config not yet set — retry on a later call.
                return "fp4"
            expert_dtype = getattr(hf_config, "expert_dtype", "fp4")
            if expert_dtype not in _DEEPSEEK_V4_EXPERT_DTYPES:
                raise ValueError(
                    f"Unsupported DeepSeek V4 expert_dtype={expert_dtype!r}; "
                    f"expected one of {_DEEPSEEK_V4_EXPERT_DTYPES}."
                )
            self._resolved_expert_dtype = expert_dtype
            from vllm.logger import init_logger

            init_logger(__name__).info_once(
                "DeepSeek V4 expert_dtype resolved to %r", expert_dtype
            )
        return self._resolved_expert_dtype

    @property
    def is_scale_e8m0(self) -> bool:
        return self.expert_dtype == "fp4"

    def _resolve_moe_overrides(self) -> None:
        if self._resolved_moe_quant_algo is not None:
            return
        try:
            hf_config = get_current_vllm_config().model_config.hf_config
        except Exception:
            return
        quant_cfg = getattr(hf_config, "quantization_config", None) or {}
        algo = (quant_cfg.get("moe_quant_algo") or "").upper() or None
        self._resolved_moe_quant_algo = algo or ""

    @property
    def moe_quant_algo(self) -> str:
        self._resolve_moe_overrides()
        return self._resolved_moe_quant_algo or ""

    def _get_nvfp4_config(self) -> ModelOptNvFp4Config:
        if self._nvfp4_config is None:
            from vllm.model_executor.layers.quantization.modelopt import (
                ModelOptNvFp4Config,
            )

            self._nvfp4_config = ModelOptNvFp4Config(
                is_checkpoint_nvfp4_serialized=True,
                kv_cache_quant_algo=None,
                exclude_modules=[],
                group_size=16,
            )
        return self._nvfp4_config

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "deepseek_v4_fp8"

    @classmethod
    def override_quantization_method(
        cls, hf_quant_cfg, user_quant, hf_config=None
    ) -> QuantizationMethods | None:
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
            if self.expert_dtype == "fp4":
                if self.moe_quant_algo == "NVFP4":
                    from vllm.model_executor.layers.quantization.modelopt import (
                        ModelOptNvFp4FusedMoE,
                    )

                    return ModelOptNvFp4FusedMoE(
                        quant_config=self._get_nvfp4_config(),
                        moe_config=layer.moe_config,
                    )
                from vllm.model_executor.layers.quantization.mxfp4 import (
                    Mxfp4MoEMethod,
                )

                return Mxfp4MoEMethod(layer.moe_config)
            # expert_dtype == "fp8": fp8 MoE method below.
            if self.is_checkpoint_fp8_serialized:
                return Fp8MoEMethod(self, layer)
            else:
                return Fp8OnlineMoEMethod(self, layer)

        if isinstance(layer, LinearBase):
            if is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                return UnquantizedLinearMethod()
            from vllm.model_executor.layers.quantization.fp8 import (
                get_marlin_input_dtype,
            )

            if not self.is_checkpoint_fp8_serialized:
                from vllm.model_executor.layers.quantization.online.fp8 import (
                    Fp8PerTensorOnlineLinearMethod,
                )

                online_method = Fp8PerTensorOnlineLinearMethod()
                online_method.marlin_input_dtype = get_marlin_input_dtype(prefix)
                return online_method
            offline_method = Fp8LinearMethod(self)
            offline_method.marlin_input_dtype = get_marlin_input_dtype(prefix)
            return offline_method

        from vllm.model_executor.layers.attention import Attention

        if isinstance(layer, Attention):
            return Fp8KVCacheMethod(self)

        return None

    def is_mxfp4_quant(self, prefix, layer):
        if not isinstance(layer, RoutedExperts) or self.expert_dtype != "fp4":
            return False
        return self.moe_quant_algo != "NVFP4"

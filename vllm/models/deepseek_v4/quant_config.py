# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Quantization config for DeepSeek V4."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.fused_moe import (
    RoutedExperts,
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4MoEMethod
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped,
)

_DEEPSEEK_V4_EXPERT_DTYPES = ("fp4", "fp8")

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptNvFp4Config,
    )


class DeepseekV4FP8Config(Fp8Config):
    """FP8 config for DeepSeek V4 with expert-dtype-aware MoE dispatch.

    DeepSeek V4 checkpoints always use FP8 block quantization for
    linear/attention layers. The MoE expert weights vary by checkpoint:
    - ``expert_dtype="fp4"`` (e.g. DeepSeek-V4-Flash): MXFP4 experts
      with ue8m0 (e8m0fnu) FP8 linear scales.
    - ``expert_dtype="fp8"`` (e.g. DeepSeek-V4-Flash-Base): FP8 block
      experts with float32 FP8 linear scales.

    The dispatch and the linear scale dtype are both keyed off
    ``expert_dtype`` from the model's hf_config; missing values default
    to ``"fp4"`` so existing FP4 checkpoints stay unchanged.

    NOTE: ``expert_dtype`` is resolved lazily because this config is
    constructed during VllmConfig setup, before ``set_current_vllm_config``
    is active. Reading hf_config eagerly in ``__init__`` would always see
    the default ``"fp4"`` and silently misroute Flash-Base checkpoints.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._resolved_expert_dtype: str | None = None
        self._resolved_moe_quant_algo: str | None = None
        self._nvfp4_config: ModelOptNvFp4Config | None = None
        # ``is_scale_e8m0`` is a property that resolves on first read,
        # by which time the current vllm_config has been set.

    @property
    def expert_dtype(self) -> str:
        if self._resolved_expert_dtype is None:
            try:
                hf_config = get_current_vllm_config().model_config.hf_config
            except Exception:
                # vllm_config not yet set; defer the decision until a
                # later call lands inside set_current_vllm_config.
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
        # FP4 checkpoints store FP8 linear scales as e8m0fnu; FP8 expert
        # checkpoints (Flash-Base) store them as float32.
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

    @staticmethod
    def _is_quark_mxfp4_ocp(hf_quant_cfg: dict) -> bool:
        """True for AMD-Quark exports whose global scheme is MXFP4."""
        weight = (hf_quant_cfg.get("global_quant_config") or {}).get("weight") or {}
        return (
            weight.get("dtype") == "fp4"
            and weight.get("qscheme") == "per_group"
            and weight.get("group_size") == 32
        )

    @classmethod
    def override_quantization_method(
        cls, hf_quant_cfg, user_quant, hf_config=None
    ) -> QuantizationMethods | None:
        if not (
            isinstance(hf_quant_cfg, dict)
            and (
                hf_quant_cfg.get("quant_method") in ("fp8", "deepseek_v4_fp8")
                or (
                    hf_quant_cfg.get("quant_method") == "quark"
                    and cls._is_quark_mxfp4_ocp(hf_quant_cfg)
                )
            )
        ):
            return None
        model_type = getattr(hf_config, "model_type", None)
        if model_type == "deepseek_v4" or user_quant == "deepseek_v4_fp8":
            return "deepseek_v4_fp8"
        return None

    @classmethod
    def from_config(cls, config: dict) -> DeepseekV4FP8Config:
        # Reroute AMD-Quark fused shared expert MXFP4 checkpoints onto the fp8
        # path: the runtime layout matches the DeepSeek-native fp8 checkpoint,
        # so translate the schema into format Fp8Config.from_config expects.
        if config.get("quant_method") == "quark":
            quark_exclude = config.get("exclude") or []
            config = {
                "quant_method": "fp8",
                "activation_scheme": "dynamic",
                "fmt": "e4m3",
                "scale_fmt": "ue8m0",
                "weight_block_size": [128, 128],
                "ignored_layers": [
                    name for name in quark_exclude if isinstance(name, str)
                ],
            }
        return cast("DeepseekV4FP8Config", super().from_config(config))

    def get_quant_method(self, layer, prefix):
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
                return Mxfp4MoEMethod(layer.moe_config)
            # expert_dtype == "fp8": fall through to Fp8Config which
            # returns Fp8MoEMethod with block-wise float32 scales.
        return super().get_quant_method(layer, prefix)

    def is_mxfp4_quant(self, prefix, layer):
        if not isinstance(layer, RoutedExperts) or self.expert_dtype != "fp4":
            return False
        return self.moe_quant_algo != "NVFP4"

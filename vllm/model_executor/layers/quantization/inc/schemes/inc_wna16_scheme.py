# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.platforms import current_platform

from ..inc_linear import INCLinearMethod
from .inc_scheme import INCScheme

if TYPE_CHECKING:
    import torch

    from ..config_parser import INCLayerConfig
    from ..inc import INCConfig

logger = init_logger(__name__)

XPU_WNA16_SUPPORTED_BITS = {2, 4}


class INCWna16Scheme(INCScheme):
    @staticmethod
    def can_handle(layer_config: "INCLayerConfig") -> bool:
        return layer_config.is_wna16_int

    def get_linear_method(
        self,
        config: "INCConfig",
        layer: "torch.nn.Module",
        prefix: str,
        layer_config: "INCLayerConfig",
    ):
        del config, layer
        if current_platform.is_xpu():
            if layer_config.bits in XPU_WNA16_SUPPORTED_BITS and layer_config.sym:
                from .inc_ark_ops import get_ark_state
                from .inc_wna16_linear import (
                    INCARKLinearMethod,
                    INCXPULinearMethod,
                )

                is_ark_available, ark_error, _, _ = get_ark_state()
                if is_ark_available:
                    return INCLinearMethod(INCARKLinearMethod(layer_config))
                if layer_config.bits == 2:
                    raise NotImplementedError(
                        "INC int2 on XPU requires the ARK backend. "
                        f"Layer: {prefix}. "
                        f"auto_round_kernel unavailable: "
                        f"{ark_error or 'unknown error'}"
                    )

                logger.debug(
                    "ARK backend is unavailable for layer %s; "
                    "falling back to the default XPU INC path. Error: %s",
                    prefix,
                    ark_error or "unknown error",
                )
                return INCLinearMethod(INCXPULinearMethod(layer_config))
            raise NotImplementedError(f"INC on XPU: unsupported config {layer_config}")

        if current_platform.is_cpu() and layer_config.is_gptq:
            if layer_config.bits == 4 and layer_config.sym:
                from .inc_ark_ops import get_ark_state
                from .inc_wna16_linear import (
                    INCARKLinearMethod,
                    INCWNA16LinearScheme,
                )

                is_ark_available, ark_error, _, _ = get_ark_state()
                if is_ark_available:
                    return INCLinearMethod(INCARKLinearMethod(layer_config))

                logger.debug(
                    "ARK backend is unavailable for layer %s; "
                    "falling back to the default CPU INC path. Error: %s",
                    prefix,
                    ark_error or "unknown error",
                )
                return INCLinearMethod(INCWNA16LinearScheme(layer_config))
            raise NotImplementedError(f"INC on CPU: unsupported config {layer_config}")

        from .inc_wna16_linear import INCWNA16LinearScheme

        return INCLinearMethod(INCWNA16LinearScheme(layer_config))

    def get_moe_method(
        self,
        config: "INCConfig",
        layer: "torch.nn.Module",
        prefix: str,
        layer_config: "INCLayerConfig",
    ):
        del config

        if (
            current_platform.is_xpu()
            and layer_config.is_gptq
            and layer_config.bits == 4
            and layer_config.sym
        ):
            from vllm.model_executor.layers.quantization.moe_wna16 import (
                MoeWNA16Config,
            )

            from .inc_ark_ops import get_ark_state
            from .inc_wna16_moe import (
                INCARKWNA16MoEMethod,
                INCWNA16MoEScheme,
            )

            is_ark_available, ark_error, _, _ = get_ark_state()
            if is_ark_available:
                moe_config = MoeWNA16Config.from_config(
                    {
                        "quant_method": "gptq",
                        "bits": layer_config.bits,
                        "group_size": layer_config.group_size,
                        "sym": layer_config.sym,
                        "lm_head": False,
                    }
                )
                return INCARKWNA16MoEMethod(moe_config, layer.moe_config)

            logger.info(
                "ARK backend is unavailable for MoE layer %s; "
                "falling back to the default WNA16 MoE path. Error: %s",
                prefix,
                ark_error or "unknown error",
            )

            return INCWNA16MoEScheme(layer_config).get_method(layer)

        from .inc_wna16_moe import INCWNA16MoEScheme

        return INCWNA16MoEScheme(layer_config).get_method(layer)

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.auto_awq import AutoAWQConfig
from vllm.model_executor.layers.quantization.auto_gptq import AutoGPTQConfig
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

from ..inc_linear import INCLinearMethod
from .inc_scheme import INCScheme

if TYPE_CHECKING:
    import torch

    from ..config_parser import INCLayerConfig
    from ..inc import INCConfig

logger = init_logger(__name__)


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
            if layer_config.bits == 4 and layer_config.sym:
                from .inc_ark_ops import get_ark_state
                from .inc_wna16_linear import (
                    INCARKLinearMethod,
                    INCXPULinearMethod,
                )

                is_ark_available, ark_error, _, _ = get_ark_state()
                if is_ark_available:
                    return INCLinearMethod(INCARKLinearMethod(layer_config))

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
        del config, prefix
        # XPU and CPU do not support MoE quantization yet
        if current_platform.is_xpu() or current_platform.is_cpu():
            from vllm.model_executor.layers.fused_moe import (
                UnquantizedFusedMoEMethod,
            )

            return UnquantizedFusedMoEMethod(layer.moe_config)
        if layer_config.is_gptq:
            return _resolve_gptq_moe(layer, layer_config)
        if layer_config.is_awq:
            return _resolve_awq_moe(layer, layer_config)
        raise NotImplementedError(f"WNA16 MoE does not support config {layer_config}")


def _resolve_gptq_moe(layer: "torch.nn.Module", layer_config: "INCLayerConfig"):
    from vllm.model_executor.layers.quantization.auto_gptq import (
        AutoGPTQMoEMethod,
    )
    from vllm.model_executor.layers.quantization.moe_wna16 import (
        MoeWNA16Config,
        MoeWNA16Method,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        check_marlin_supported,
        check_moe_marlin_supports_layer,
    )

    group_size = layer_config.group_size
    if not isinstance(group_size, int):
        raise ValueError(
            f"INC WNA16 MoE requires scalar group_size, but found {group_size!r}."
        )

    gptq_type_map = {
        (4, True): scalar_types.uint4b8,
        (8, True): scalar_types.uint8b128,
    }
    use_marlin = (layer_config.bits, layer_config.sym) in gptq_type_map
    if use_marlin:
        use_marlin = check_marlin_supported(
            gptq_type_map[(layer_config.bits, layer_config.sym)],
            group_size,
            has_zp=not layer_config.sym,
        ) and check_moe_marlin_supports_layer(layer, group_size)

    if use_marlin:
        return AutoGPTQMoEMethod(
            AutoGPTQConfig(
                weight_bits=layer_config.bits,
                group_size=group_size,
                desc_act=False,
                is_sym=layer_config.sym,
                lm_head_quantized=False,
                dynamic={},
                full_config={},
            ),
            layer.moe_config,
        )

    moe_config = MoeWNA16Config.from_config(
        {
            "quant_method": "gptq",
            "bits": layer_config.bits,
            "group_size": group_size,
            "sym": layer_config.sym,
            "lm_head": False,
        }
    )
    return MoeWNA16Method(moe_config, layer.moe_config)


def _resolve_awq_moe(layer: "torch.nn.Module", layer_config: "INCLayerConfig"):
    from vllm.model_executor.layers.quantization.auto_awq import AutoAWQMoEMethod
    from vllm.model_executor.layers.quantization.moe_wna16 import (
        MoeWNA16Config,
        MoeWNA16Method,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        check_marlin_supported,
        check_moe_marlin_supports_layer,
    )

    group_size = layer_config.group_size
    if not isinstance(group_size, int):
        raise ValueError(
            f"INC WNA16 MoE requires scalar group_size, but found {group_size!r}."
        )

    awq_type_map = {
        4: scalar_types.uint4,
        8: scalar_types.uint8,
    }
    use_marlin = layer_config.bits in awq_type_map
    if use_marlin:
        use_marlin = check_marlin_supported(
            awq_type_map[layer_config.bits],
            group_size,
            not layer_config.sym,
        ) and check_moe_marlin_supports_layer(layer, group_size)

    if use_marlin:
        return AutoAWQMoEMethod(
            AutoAWQConfig(
                weight_bits=layer_config.bits,
                group_size=group_size,
                zero_point=not layer_config.sym,
                lm_head_quantized=False,
                modules_to_not_convert=[],
                full_config={},
            ),
            layer.moe_config,
        )

    moe_config = MoeWNA16Config.from_config(
        {
            "quant_method": "awq",
            "bits": layer_config.bits,
            "group_size": group_size,
            "zero_point": not layer_config.sym,
            "lm_head": False,
        }
    )
    return MoeWNA16Method(moe_config, layer.moe_config)

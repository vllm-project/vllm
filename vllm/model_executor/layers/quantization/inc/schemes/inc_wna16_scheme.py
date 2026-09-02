# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

from vllm import envs
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.auto_awq import AutoAWQConfig
from vllm.model_executor.layers.quantization.auto_gptq import AutoGPTQConfig
from vllm.platforms import current_platform

from ..inc_linear import INCLinearMethod
from .inc_scheme import INCScheme

if TYPE_CHECKING:
    import torch

    from ..config_parser import INCLayerConfig
    from ..inc import INCConfig

logger = init_logger(__name__)

XPU_WNA16_SUPPORTED_BITS = {2, 4}

# Backends selectable through VLLM_XPU_INC_WNA16_BACKEND that are served by the
# oneDNN int4 GEMMs rather than by ARK. These only cover int4.
XPU_ONEDNN_BACKENDS = ("w4a16", "w4a8")


def _check_xpu_w4a8_supported(layer_config: "INCLayerConfig", prefix: str) -> None:
    """Raise unless ``int4_gemm_w4a8`` can serve this layer.

    The backend is requested explicitly, so an unusable configuration is an
    error rather than something to silently fall back from.
    """
    import torch

    if not hasattr(torch.ops._xpu_C, "int4_gemm_w4a8"):
        raise NotImplementedError(
            "VLLM_XPU_INC_WNA16_BACKEND=w4a8 requires the int4_gemm_w4a8 op, "
            "which this build of vllm-xpu-kernels does not provide. "
            f"Layer: {prefix}."
        )
    assert isinstance(layer_config.group_size, int), (
        "WNA16 only supports integer group_size."
    )
    if layer_config.group_size <= 0 or layer_config.group_size % 32 != 0:
        raise NotImplementedError(
            "VLLM_XPU_INC_WNA16_BACKEND=w4a8 requires a group size that is a "
            f"positive multiple of 32, got {layer_config.group_size}. "
            f"Layer: {prefix}."
        )


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
                from .inc_w4a8_linear import INCXPUW4A8LinearMethod
                from .inc_wna16_linear import (
                    INCARKLinearMethod,
                    INCXPULinearMethod,
                )

                backend = envs.VLLM_XPU_INC_WNA16_BACKEND
                if backend in XPU_ONEDNN_BACKENDS:
                    if layer_config.bits != 4:
                        raise NotImplementedError(
                            f"VLLM_XPU_INC_WNA16_BACKEND={backend} only supports "
                            f"int4, got int{layer_config.bits}. Layer: {prefix}."
                        )
                    if backend == "w4a8":
                        _check_xpu_w4a8_supported(layer_config, prefix)
                        return INCLinearMethod(INCXPUW4A8LinearMethod(layer_config))
                    return INCLinearMethod(INCXPULinearMethod(layer_config))

                is_ark_available, ark_error, _, _ = get_ark_state()
                if backend == "ark" and not is_ark_available:
                    raise NotImplementedError(
                        "VLLM_XPU_INC_WNA16_BACKEND=ark was requested but "
                        f"auto_round_kernel is unavailable: "
                        f"{ark_error or 'unknown error'}. Layer: {prefix}."
                    )
                if is_ark_available:
                    return INCLinearMethod(INCARKLinearMethod(layer_config))
                elif layer_config.bits == 2:
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
        del config, prefix
        # CPU does not support quantized MoE yet.
        if current_platform.is_cpu():
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
        check_moe_marlin_supports_layer,
    )

    assert isinstance(layer_config.group_size, int), (
        "WNA16 only supports integer group_size."
    )

    # AutoGPTQMoEMethod selects its fused-MoE backend through the WNA16 oracle
    # (Marlin on CUDA, XPUExpertsWNA16 on XPU). Gate only on the layer-shape
    # check like compressed-tensors does; the capability-based
    # check_marlin_supported is skipped so the XPU path is reachable.
    use_marlin = (layer_config.bits, layer_config.sym) in {
        (4, True),
        (8, True),
    } and check_moe_marlin_supports_layer(layer, layer_config.group_size)

    if use_marlin:
        return AutoGPTQMoEMethod(
            AutoGPTQConfig(
                weight_bits=layer_config.bits,
                group_size=layer_config.group_size,
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
            "group_size": layer_config.group_size,
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
        check_moe_marlin_supports_layer,
    )

    assert isinstance(layer_config.group_size, int), (
        "WNA16 only supports integer group_size."
    )

    use_marlin = layer_config.bits in (4, 8) and check_moe_marlin_supports_layer(
        layer, layer_config.group_size
    )

    if use_marlin:
        return AutoAWQMoEMethod(
            AutoAWQConfig(
                weight_bits=layer_config.bits,
                group_size=layer_config.group_size,
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
            "group_size": layer_config.group_size,
            "zero_point": not layer_config.sym,
            "lm_head": False,
        }
    )
    return MoeWNA16Method(moe_config, layer.moe_config)

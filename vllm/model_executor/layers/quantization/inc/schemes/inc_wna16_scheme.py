# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

from vllm import envs
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
# On CUDA the Marlin/GPTQ/AWQ kernels only cover 4/8-bit. The remaining
# widths (2/3/5/6/7) have no dedicated CUDA kernel, so they are dispatched to
# the humming kernel instead. This mirrors compressed-tensors WNA16, whose
# 5/6/7-bit biased scalar types also fall back to humming at kernel selection.
CUDA_HUMMING_SUPPORTED_BITS = {2, 3, 5, 6, 7}

# Backends selectable through VLLM_XPU_INC_WNA16_BACKEND that are served by the
# oneDNN int4 GEMMs rather than by ARK. These only cover int4.
XPU_ONEDNN_BACKENDS = ("w4a16", "w4a8")


def _check_xpu_w4a8_supported(layer_config: "INCLayerConfig", prefix: str) -> None:
    """Raise unless ``int4_gemm_w4a8`` can serve this linear layer.

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

        # CUDA low-bit (2/3/5/6/7): no Marlin/GPTQ/AWQ kernel, route to humming
        # so a single model can mix 4/8-bit (Marlin) and 2/3/5/6/7-bit (humming)
        # layers.
        if (
            current_platform.is_cuda()
            and layer_config.bits in CUDA_HUMMING_SUPPORTED_BITS
        ):
            return _build_humming_linear_method(layer_config)

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
            from .inc_ark_ops import get_ark_state
            from .inc_wna16_moe import INCARKWNA16MoEMethod

            backend = envs.VLLM_XPU_INC_WNA16_BACKEND
            if backend != "w4a16":
                from vllm.model_executor.layers.quantization.moe_wna16 import (
                    MoeWNA16Config,
                )

                group_size = layer_config.group_size
                assert isinstance(group_size, int), (
                    "WNA16 only supports integer group_size."
                )

                def make_moe_config() -> MoeWNA16Config:
                    return MoeWNA16Config.from_config(
                        {
                            "quant_method": "gptq",
                            "bits": layer_config.bits,
                            "group_size": group_size,
                            "sym": layer_config.sym,
                            "lm_head": False,
                        }
                    )

                is_ark_available, ark_error, ark, _ = get_ark_state()
                xpu_lib = getattr(ark, "xpu_lib", None) if ark is not None else None
                ark_moe_error = ark_error or "ARK MoE kernels are unavailable"

                if backend in ("w4a8", "ark"):
                    from .inc_w4a8_moe import (
                        INCARKW4A8MoEMethod,
                        check_xpu_moe_w4a8_supported,
                        has_ark_w4a8_moe_kernel,
                    )

                    is_ark_w4a8_moe_available = has_ark_w4a8_moe_kernel(
                        is_ark_available,
                        ark,
                    )
                    if not is_ark_w4a8_moe_available:
                        if backend == "w4a8":
                            raise NotImplementedError(
                                "VLLM_XPU_INC_WNA16_BACKEND=w4a8 was requested but "
                                "ARK W4A8 prefill/W4A16 decode MoE kernels are "
                                f"unavailable: {ark_moe_error}. Layer: {prefix}."
                            )
                        logger.debug(
                            "ARK W4A8 MoE kernels are unavailable for layer %s; "
                            "falling back to ARK WNA16 MoE. Error: %s",
                            prefix,
                            ark_moe_error,
                        )
                    else:
                        try:
                            check_xpu_moe_w4a8_supported(
                                layer,
                                layer_config,
                                prefix,
                            )
                        except NotImplementedError as exc:
                            if backend == "w4a8":
                                raise
                            logger.debug(
                                "ARK W4A8 MoE is unsupported for layer %s; "
                                "falling back to ARK WNA16 MoE. Error: %s",
                                prefix,
                                exc,
                            )
                        else:
                            return INCARKW4A8MoEMethod(
                                make_moe_config(),
                                layer.moe_config,
                            )

                is_ark_moe_available = (
                    is_ark_available
                    and ark is not None
                    and xpu_lib is not None
                    and hasattr(ark, "MoeSymmetricGemm")
                )
                if backend == "ark" and not is_ark_moe_available:
                    raise NotImplementedError(
                        "VLLM_XPU_INC_WNA16_BACKEND=ark was requested but "
                        f"ARK MoE kernels are unavailable: {ark_moe_error}. "
                        f"Layer: {prefix}."
                    )

                if is_ark_moe_available:
                    return INCARKWNA16MoEMethod(
                        make_moe_config(),
                        layer.moe_config,
                    )

                logger.info(
                    "ARK backend is unavailable for MoE layer %s; "
                    "falling back to the default WNA16 MoE path. Error: %s",
                    prefix,
                    ark_moe_error,
                )

        # CUDA low-bit (2/3/5/6/7): route to the humming MoE kernel (see above).
        if (
            current_platform.is_cuda()
            and layer_config.bits in CUDA_HUMMING_SUPPORTED_BITS
        ):
            return _build_humming_moe_method(layer, layer_config)

        from .inc_wna16_moe import INCWNA16MoEScheme

        return INCWNA16MoEScheme(layer_config).get_method(layer)


def _humming_weight_config(layer_config: "INCLayerConfig") -> dict:
    """Build the humming weight-schema config for a WNA16 int checkpoint."""
    if layer_config.is_gptq:
        return {
            "quant_method": "gptq",
            "bits": layer_config.bits,
            "group_size": layer_config.group_size,
            "desc_act": False,
            "sym": layer_config.sym,
        }
    if layer_config.is_awq:
        return {
            "quant_method": "awq",
            "bits": layer_config.bits,
            "group_size": layer_config.group_size,
            "zero_point": not layer_config.sym,
        }
    raise NotImplementedError(
        "INC humming dispatch only supports gptq/awq packed int checkpoints, "
        f"but found {layer_config}."
    )


def _build_humming_quant_config(layer_config: "INCLayerConfig"):
    from vllm.model_executor.layers.quantization.humming import (
        HummingLayerQuantizationConfig,
    )
    from vllm.utils.humming import BaseWeightSchema

    weight_schema = BaseWeightSchema.from_config(_humming_weight_config(layer_config))
    return HummingLayerQuantizationConfig(weight_schema=weight_schema)


def _build_humming_linear_method(layer_config: "INCLayerConfig"):
    from vllm.model_executor.layers.quantization.humming import HummingLinearMethod

    return HummingLinearMethod(_build_humming_quant_config(layer_config))


def _build_humming_moe_method(layer: "torch.nn.Module", layer_config: "INCLayerConfig"):
    from vllm.model_executor.layers.quantization.humming import HummingMoEMethod

    return HummingMoEMethod(_build_humming_quant_config(layer_config), layer.moe_config)

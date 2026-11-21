# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm import envs
from vllm.logger import init_logger
from vllm.model_executor.kernels.linear.nvfp4.base import NvFp4LinearLayerConfig
from vllm.model_executor.kernels.linear.nvfp4.marlin import MarlinNvFp4LinearKernel
from vllm.model_executor.layers.fusion.quant_activation import expose_input_quant_key
from vllm.model_executor.layers.quantization.modelopt import ModelOptLinearMethod
from vllm.model_executor.layers.quantization.utils.quant_utils import kNvfp4Static
from vllm.model_executor.utils import is_weights_pre_processed
from vllm.platforms import current_platform

logger = init_logger(__name__)


def select_mamba_marlin_kernels(layer, config) -> None:
    """Select Marlin before packing eligible Mamba2 projection weights."""
    if (
        not current_platform.is_cuda()
        or envs.VLLM_BATCH_INVARIANT
        or config.lora_config is not None
        or is_weights_pre_processed()
    ):
        return
    capability = current_platform.get_device_capability()
    if capability is None or capability.to_int() not in (120, 121):
        return
    if MarlinNvFp4LinearKernel.__name__ in envs.VLLM_DISABLED_KERNELS:
        return
    supported, _ = MarlinNvFp4LinearKernel.is_supported(capability.to_int())
    if not supported:
        return
    for projection in (layer.in_proj, layer.out_proj):
        method = projection.quant_method
        previous_kernel = getattr(method, "kernel", None)
        if type(previous_kernel) is MarlinNvFp4LinearKernel:
            continue
        if not (
            type(method) is ModelOptLinearMethod
            and method.spec.weight == kNvfp4Static
            and method.spec.activation is None
            and method.akey is None
            and projection.params_dtype in (torch.float16, torch.bfloat16)
        ):
            continue
        previous_config = getattr(previous_kernel, "config", None)
        if not (
            isinstance(previous_config, NvFp4LinearLayerConfig)
            and MarlinNvFp4LinearKernel.can_implement(previous_config)[0]
        ):
            continue
        method.kernel = MarlinNvFp4LinearKernel(previous_config)
        if hasattr(projection, "_input_quant_key"):
            del projection._input_quant_key
        expose_input_quant_key(projection, method.kernel)
        logger.info_once(
            "Using Marlin instead of %s for Mamba2 projections on SM%d; "
            "this takes precedence over --linear-backend for these projections.",
            type(previous_kernel).__name__,
            capability.to_int(),
        )

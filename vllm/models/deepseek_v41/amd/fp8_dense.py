# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load 1017 MXFP8 dense weights for the block-FP8 expert conversion on gfx942."""

import torch

from vllm import envs
from vllm.config import get_current_vllm_config_or_none
from vllm.model_executor.kernels.linear.mxfp8.emulation import (
    EmulationMxfp8LinearKernel,
)
from vllm.model_executor.layers.linear import (
    UnquantizedLinearMethod,
    register_weight_loader_v2_supported_method,
)
from vllm.model_executor.layers.quantization.modelopt import ModelOptLinearMethod
from vllm.model_executor.model_loader.reload.direct import direct_reload_enabled
from vllm.model_executor.utils import is_weights_pre_processed
from vllm.platforms.rocm import on_gfx942


@register_weight_loader_v2_supported_method
class Bf16EmulatedLinearMethod(ModelOptLinearMethod):
    @staticmethod
    def matches(online) -> bool:
        cfg = get_current_vllm_config_or_none()
        if cfg is None or cfg.model_config is None:
            return False
        text_config = cfg.model_config.hf_text_config
        return (
            envs.VLLM_MXFP8_EMULATION_DEQUANT_AT_LOAD
            and cfg.model_config.dtype == torch.bfloat16
            and text_config.model_type in ("deepseek_v41", "deepseek_v41_text")
            and text_config.hidden_size == 5120
            and text_config.moe_intermediate_size == 2304
            and cfg.kernel_config.moe_backend == "aiter"
            and not cfg.kernel_config.monokernel
            and cfg.parallel_config.tensor_parallel_size == 8
            and cfg.parallel_config.data_parallel_size == 1
            and not cfg.parallel_config.enable_expert_parallel
            and cfg.weight_transfer_config is None
            and not direct_reload_enabled()
            and not is_weights_pre_processed()
            and cfg.lora_config is None
            and not online.args.ignore
            and online.args.targets == {"*ffn.experts*": "fp8_per_block"}
            and on_gfx942()
        )

    def process_weights_after_loading(self, layer) -> None:
        if not hasattr(layer, "weight_scale"):
            # An ordinary second finalization sees the already converted layer.
            UnquantizedLinearMethod.process_weights_after_loading(
                layer.quant_method, layer
            )
            return
        # Preserve checkpoint scale expansion, fused shard loading and TP slicing.
        super().process_weights_after_loading(layer)
        if is_weights_pre_processed() or not isinstance(
            self.kernel, EmulationMxfp8LinearKernel
        ):
            return
        # Emulation has already materialized BF16. Use the offline checkpoint's
        # compute method, including its ROCm dispatcher and activation contract.
        if layer.weight.dtype != torch.bfloat16:
            raise RuntimeError("Expected BF16 weights after MXFP8 emulation load")
        method = UnquantizedLinearMethod()
        method.process_weights_after_loading(layer)
        del layer.weight_scale
        # Layerwise reload restores the original MXFP8 tensors and their
        # loaders, then calls the active method's post-load hook again.
        method.process_weights_after_loading = (  # type: ignore[method-assign]
            self.process_weights_after_loading
        )
        layer.quant_method = method

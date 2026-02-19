# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.platforms import current_platform

logger = init_logger(__name__)


class BaseKVCacheMethod(QuantizeMethodBase):
    """
    Quant method that adds `_k_scale` and `_v_scale` attributes to the
    Attention layer to support loading those scaling factors from checkpoints.
    The k/v_scale will be used to:
        - quantize k/v_cache entries before saving them to the cache
        - dequantize k/v_cache entries before fetching them from the cache

    :param quant_config: the appropriate QuantizationConfig
    """

    def __init__(self, quant_config: QuantizationConfig):
        self.quant_config = quant_config

    def create_weights(self, layer: torch.nn.Module):
        """
        Create "weight" (aka q_scale, k_scale and v_scale)
        for an attention layer.
        """
        # Initialize the Q and KV cache scales to -1.0, an invalid value.
        # If the q and k/v_scales appear in the checkpoint, it will be
        # overwritten when loading weights.
        layer.q_scale = torch.nn.Parameter(torch.tensor(-1.0), requires_grad=False)
        layer.k_scale = torch.nn.Parameter(torch.tensor(-1.0), requires_grad=False)
        layer.v_scale = torch.nn.Parameter(torch.tensor(-1.0), requires_grad=False)
        # Initialize P = softmax(QK^T) scales
        layer.prob_scale = torch.nn.Parameter(torch.tensor(-1.0), requires_grad=False)

    def apply(self, layer: torch.nn.Module) -> torch.Tensor:
        raise RuntimeError(f"{self.__class__.__name__}.apply should not be called.")

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # If the kv-cache dtype is auto, we enforce the k/v_scale to be 1.0
        # regardless whether the kv-scale is available in the checkpoint.
        # No need to process kv scales after loading if we are going to
        # calculate them on the fly.
        if layer.kv_cache_dtype != "auto" and not layer.calculate_kv_scales:
            if layer.k_scale > 0.0 and layer.v_scale > 0.0:
                # We prefer to use separate k_scale and v_scale if present
                k_scale = layer.k_scale.to("cpu").tolist()
                v_scale = layer.v_scale.to("cpu").tolist()
                if current_platform.is_fp8_fnuz():
                    k_scale *= 2
                    v_scale *= 2
            elif layer.k_scale < 0.0 and layer.v_scale < 0.0:
                # If no scales were loaded (both scales are invalid negative
                # values), use the default value of 1.0
                k_scale = 1.0
                v_scale = 1.0
            else:
                # If we find a single kv_scale in the checkpoint, we remap
                # kv_scale to k_scale during weight loading, and duplicate
                # k_scale to v_scale here
                assert layer.k_scale > 0.0
                scale_to_duplicate = max(layer.k_scale, layer.v_scale)
                k_scale = scale_to_duplicate.to("cpu").tolist()
                v_scale = scale_to_duplicate.to("cpu").tolist()
                if current_platform.is_fp8_fnuz():
                    k_scale *= 2
                    v_scale *= 2

            if not isinstance(k_scale, float) or not isinstance(v_scale, float):
                raise ValueError(
                    "Only support per-tensor scaling factor for fp8 KV cache"
                )

            if layer.q_scale < 0.0:
                logger.warning_once(
                    "Checkpoint does not provide a q scaling factor. "
                    "Setting it to k_scale. This only matters for "
                    "FP8 Attention backends (flash-attn or flashinfer)."
                )
                layer._q_scale.copy_(k_scale)
                layer._q_scale_float = k_scale

            # These are used in the final Attention.forward()
            layer._k_scale.copy_(k_scale)
            layer._v_scale.copy_(v_scale)
            layer._k_scale_float = k_scale
            layer._v_scale_float = v_scale
            if k_scale == 1.0 and v_scale == 1.0 and "e5m2" not in layer.kv_cache_dtype:
                logger.warning_once(
                    "Using KV cache scaling factor 1.0 for fp8_e4m3. "
                    "If this is unintended, verify that k/v_scale "
                    "scaling factors are properly set in the checkpoint."
                )

        if layer.q_scale > 0.0:
            q_scale = layer.q_scale
            if current_platform.is_fp8_fnuz():
                q_scale *= 2
            layer.calculate_kv_scales = False
        else:
            q_scale = 1.0
        if layer.prob_scale > 0.0:
            prob_scale = layer.prob_scale
            if current_platform.is_fp8_fnuz():
                prob_scale *= 2
        else:
            prob_scale = 1.0

        is_singleton_float = (
            lambda x: isinstance(x, float)
            or isinstance(x, torch.Tensor)
            and x.numel() == 1
            and x.is_floating_point()
        )
        if not is_singleton_float(q_scale) or not is_singleton_float(prob_scale):
            raise ValueError(
                "Only support per-tensor scaling factorfor fp8-quantized Q/prob"
            )

        # These are used in the final Attention.forward()
        layer._q_scale.copy_(q_scale)
        layer._q_scale_float = (
            q_scale.item() if isinstance(q_scale, torch.Tensor) else q_scale
        )

        layer._prob_scale.copy_(prob_scale)
        if layer.kv_cache_dtype == "fp8" and (q_scale == 1.0 or prob_scale == 1.0):
            logger.warning_once(
                f"Using uncalibrated q_scale {q_scale} and/or prob_scale "
                f"{prob_scale} with fp8 attention. This may cause accuracy "
                "issues. Please make sure q/prob scaling factors are "
                "available in the fp8 checkpoint."
            )

        del layer.k_scale
        del layer.v_scale
        del layer.q_scale
        del layer.prob_scale

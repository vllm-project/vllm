# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.platforms import current_platform
from vllm.v1.attention.backend import is_quantized_kv_cache

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
        # skip if there are no weights to process (for example, weight reloading)
        if not hasattr(layer, "q_scale"):
            assert not hasattr(layer, "k_scale")
            assert not hasattr(layer, "v_scale")
            assert not hasattr(layer, "prob_scale")
            return

        # If the kv-cache is not quantized, we enforce the k/v_scale to be 1.0
        # regardless whether the kv-scale is available in the checkpoint.
        # No need to process kv scales after loading if we are going to
        # calculate them on the fly.
        if (
            is_quantized_kv_cache(layer.kv_cache_dtype)
            and not layer.calculate_kv_scales
        ):
            is_int8 = layer.kv_cache_dtype.startswith("int8")
            is_fp8 = layer.kv_cache_dtype.startswith("fp8")

            # Use .all() so the comparisons work for both scalar and per-head
            # [num_kv_heads] tensors (a tensor in a bool context raises an error).
            k_scale_all_positive = bool((layer.k_scale > 0.0).all())
            v_scale_all_positive = bool((layer.v_scale > 0.0).all())
            k_scale_all_negative = bool((layer.k_scale < 0.0).all())
            v_scale_all_negative = bool((layer.v_scale < 0.0).all())

            if k_scale_all_positive and v_scale_all_positive:
                # We prefer to use separate k_scale and v_scale if present.
                # INT8 may carry per-head tensors ([num_kv_heads]); FP8 is
                # always per-tensor (scalar float).
                if is_int8 and layer.k_scale.numel() > 1:
                    # Per-head INT8 scale — keep as CPU float32 tensor
                    k_scale = layer.k_scale.to("cpu").to(torch.float32)
                    v_scale = layer.v_scale.to("cpu").to(torch.float32)
                else:
                    k_scale = layer.k_scale.to("cpu").tolist()
                    v_scale = layer.v_scale.to("cpu").tolist()
                    # FNUZ fp32→fp8 range doubles the scale; INT8 unaffected
                    if current_platform.is_fp8_fnuz() and is_fp8:
                        k_scale *= 2
                        v_scale *= 2
            elif k_scale_all_negative and v_scale_all_negative:
                # No scales in checkpoint.
                if is_int8:
                    # INT8 with scale=1.0 is catastrophic: typical attention
                    # values (e.g. ±0.1–0.5) all round to 0 in int8, destroying
                    # all KV cache information and producing garbage output.
                    # Automatically enable dynamic scale computation instead.
                    logger.warning_once(
                        "INT8 KV cache: no calibrated k/v_scale found in "
                        "checkpoint. Automatically enabling --calculate-kv-scales "
                        "to compute scales from the first batch. For best accuracy, "
                        "use a checkpoint with pre-calibrated k/v_scale values "
                        "(scale = absmax(tensor) / 127 for INT8)."
                    )
                    layer.calculate_kv_scales = True
                    # Exit early — dynamic computation will set _k_scale/_v_scale
                    # on the first forward pass via calc_kv_scales().
                    del layer.k_scale
                    del layer.v_scale
                    del layer.q_scale
                    del layer.prob_scale
                    return
                # FP8 fall back to 1.0 (less catastrophic than INT8)
                k_scale = 1.0
                v_scale = 1.0
            else:
                # Single shared kv_scale in checkpoint — duplicate to k and v.
                # Per-head shared scales are not supported in this path.
                assert k_scale_all_positive
                scale_to_duplicate = max(layer.k_scale, layer.v_scale)
                k_scale = scale_to_duplicate.to("cpu").tolist()
                v_scale = scale_to_duplicate.to("cpu").tolist()
                # Only apply FNUZ scaling for FP8, not INT8
                if current_platform.is_fp8_fnuz() and is_fp8:
                    k_scale *= 2
                    v_scale *= 2

            # FP8 must always be per-tensor (scalar float)
            if is_fp8 and (
                not isinstance(k_scale, float) or not isinstance(v_scale, float)
            ):
                raise ValueError(
                    "Only support per-tensor scaling factor for fp8 KV cache"
                )

            if layer.q_scale < 0.0:
                logger.warning_once(
                    "Checkpoint does not provide a q scaling factor. "
                    "Setting it to k_scale. This only matters for "
                    "FP8 Attention backends (flash-attn or flashinfer)."
                )
                layer._q_scale.copy_(k_scale if isinstance(k_scale, float) else 1.0)
                layer._q_scale_float = k_scale if isinstance(k_scale, float) else 1.0

            # Store scales for use in Attention.forward().
            # INT8 per-head: replace the scalar buffer with the per-head tensor
            # so the kernel can index into it by kv_head_idx.
            if isinstance(k_scale, torch.Tensor) and k_scale.numel() > 1:
                # Per-head INT8: replace scalar buffer with [num_kv_heads] tensor.
                # _k_scale_float is set to 1.0 as a safe placeholder; the triton
                # backend uses _k_scale (tensor) directly, and other backends do
                # not support INT8 KV cache.
                device = layer._k_scale.device
                layer._k_scale = k_scale.to(device)
                layer._v_scale = v_scale.to(device)
                layer._k_scale_float = 1.0
                layer._v_scale_float = 1.0
            else:
                layer._k_scale.copy_(k_scale)
                layer._v_scale.copy_(v_scale)
                layer._k_scale_float = float(k_scale)
                layer._v_scale_float = float(v_scale)

            # Warn about INT8 scale=1.0 (very likely to cause overflow)
            if (
                is_int8
                and isinstance(k_scale, float)
                and (k_scale == 1.0 or v_scale == 1.0)
            ):
                logger.warning_once(
                    f"INT8 KV cache using scale k={k_scale:.6f}, "
                    f"v={v_scale:.6f}. "
                    "Scale=1.0 is likely incorrect for INT8 and "
                    "will cause accuracy issues. "
                    "Expected: scale = absmax(tensor) / 127 for INT8. "
                    "This will result in value overflow and severe precision loss."
                )

            # Warn about FP8 e4m3 with scale=1.0 (less critical, informational)
            if (
                is_fp8
                and k_scale == 1.0
                and v_scale == 1.0
                and "e5m2" not in layer.kv_cache_dtype
            ):
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

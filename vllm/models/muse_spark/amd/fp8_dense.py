# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP8 W8A8 dense linears for Muse Spark's target decoder on ROCm gfx950.

The online `fp8_per_channel` quantization (`--quantization-config`) of the target
model's fused qkv+o-gate (`qkv_proj`), MoE `pooler` and `unpooler` runs on the
`ops.fp8_linear` GEMMs instead of the generic FP8 linear kernels.

Weights are quantized as `fp8_per_channel` does (FP8 e4m3, one FP32 scale per output
channel) but kept in `[N, K]` layout. Activations are quantized per token (`amax /
448` of the BF16-rounded row) by the norm that precedes each GEMM instead of by a
separate kernel:

* pre-qkv norm: FP8 and scales only;
* pre-FFN norm: BF16 (the router needs it), FP8 and scales;
* unpooler: its input is the experts' BF16 output (W8A16, or per-token quant plus
  W8A8, by M).
"""

import torch

from vllm import envs
from vllm.config import get_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.online.base import (
    OnlineQuantizationConfig,
)
from vllm.model_executor.layers.quantization.online.fp8 import (
    Fp8PtpcOnlineLinearMethod,
    _fp8_channel_scale,
    _fp8_quant_per_channel,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import weight_amax
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform

from .ops import fp8_linear

logger = init_logger(__name__)

# Quick all-reduce levels whose error on top of the FP8 GEMMs costs too much accuracy.
_UNSUPPORTED_QUICK_REDUCE = ("INT4", "INT3")


def online_quant_config(
    quant_config: QuantizationConfig | None,
) -> OnlineQuantizationConfig | None:
    """The online part of `quant_config`, for layers the checkpoint keeps BF16."""
    if isinstance(quant_config, OnlineQuantizationConfig):
        return quant_config
    return getattr(quant_config, "online_quantization_config", None)


def online_target(layer: LinearBase, quant_config: QuantizationConfig | None) -> bool:
    """Whether an online quantization target applies to `layer`."""
    online = online_quant_config(quant_config)
    return (
        online is not None
        and online.resolve_quant_method_cls(type(layer), layer.prefix) is not None
    )


def _on_gfx950() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx950

    return on_gfx950()


def use_muse_kernels(
    layer: LinearBase, quant_config: QuantizationConfig | None
) -> bool:
    """Run an online `fp8_per_channel` layer on the Muse FP8 kernels.

    Returns:
        Whether the layer is FP8 now; False if no online target applies to it.

    Raises:
        NotImplementedError: For another online target (including a Hadamard
            wrapper), a requantized checkpoint layer, or off gfx950 / without
            AITER.
        ValueError: With LoRA (the FP8 paths call the base GEMMs directly),
            `--monokernel` (it reads the BF16 weights), or a coarse quick
            all-reduce (INT4 / INT3), whose quantization error on top of FP8 GEMMs
            costs too much accuracy.

    """
    if not online_target(layer, quant_config):
        return False
    method = layer.quant_method
    if type(method) is not Fp8PtpcOnlineLinearMethod or (
        method.requantization_source is not None
    ):
        raise NotImplementedError(
            f"Muse Spark supports only the 'fp8_per_channel' online quantization "
            f"of BF16 weights on {layer.prefix}; got {type(method).__name__}."
        )
    from vllm._aiter_ops import is_aiter_found_and_supported

    if not (_on_gfx950() and is_aiter_found_and_supported()):
        raise NotImplementedError(
            "Muse Spark FP8 dense linears need ROCm gfx950 with AITER."
        )
    vllm_config = get_current_vllm_config()
    if vllm_config.lora_config is not None:
        raise ValueError(
            "Muse Spark FP8 dense linears do not support LoRA: the FP8 paths call "
            "the base layers' GEMMs directly."
        )
    if vllm_config.kernel_config.monokernel:
        raise ValueError("Muse Spark FP8 dense linears do not support --monokernel.")
    level = envs.VLLM_ROCM_QUICK_REDUCE_QUANTIZATION
    if level in _UNSUPPORTED_QUICK_REDUCE:
        raise ValueError(
            f"Muse Spark FP8 dense linears do not support "
            f"VLLM_ROCM_QUICK_REDUCE_QUANTIZATION={level}: INT4 / INT3 all-reduce "
            "quantization on top of FP8 GEMMs collapses accuracy. Use INT8, FP or "
            "NONE."
        )
    layer.quant_method = Fp8DenseLinearMethod()
    return True


def is_converted(layer: torch.nn.Module | None) -> bool:
    return isinstance(getattr(layer, "quant_method", None), Fp8DenseLinearMethod)


class Fp8DenseLinearMethod(Fp8PtpcOnlineLinearMethod):
    """`fp8_per_channel` in the Muse kernels' layout: `weight` `[N, K]` e4m3 and
    `weight_scale` `[N]` FP32. A BF16 input runs W8A16, or quant + W8A8 (the paths
    that do not come through a fused FP8 producer, such as the unpooler).
    """

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return
        weight = layer.weight
        scale = _fp8_channel_scale(weight_amax(weight, dim=-1, keepdim=True))
        replace_parameter(layer, "weight", _fp8_quant_per_channel(weight, scale))
        replace_parameter(layer, "weight_scale", scale.view(-1))
        layer.input_scale = None
        layer._already_called_process_weights_after_loading = True
        n, k = layer.weight.shape
        logger.info_once(
            "Muse Spark FP8 dense [%d, %d] GEMMs: %s.",
            n,
            k,
            fp8_linear.routes(n, k),
            scope="local",
        )

    def apply(self, layer, x, bias=None):
        assert bias is None
        return fp8_linear.linear(layer, x)


def full_weight_parts(layer: torch.nn.Module, group, world: int):
    """All-gather the FP8 shards and their row scales: `([wq_r], [ws_r])` in rank
    order.
    """
    wq = layer.weight.data
    ws = layer.weight_scale.data
    wparts = [torch.empty_like(wq.view(torch.uint8)) for _ in range(world)]
    torch.distributed.all_gather(wparts, wq.view(torch.uint8), group=group)
    sparts = [torch.empty_like(ws) for _ in range(world)]
    torch.distributed.all_gather(sparts, ws, group=group)
    return [p.view(fp8_linear.F8) for p in wparts], sparts

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP8 W8A8 dense linears for Muse Spark's target decoder on ROCm gfx950.

The online `fp8_per_channel` quantization (`--quantization-config`) of the target
model's fused qkv+o-gate (`qkv_proj`), MoE `pooler` and `unpooler` runs on vLLM's
FP8 linear kernels. Where the selected kernel takes pre-quantized per-token FP8
input (`QuantizedActivation`), the norm that precedes the GEMM quantizes it instead
of a separate kernel:

* pre-qkv norm: FP8 and scales only;
* pre-FFN norm: BF16 (the router needs it), FP8 and scales;
* unpooler: its input is the experts' BF16 output, which the kernel quantizes.
"""

import torch

from vllm import envs
from vllm.config import get_current_vllm_config
from vllm.model_executor.kernels.linear import (
    AiterPerTokenFp8ScaledMMLinearKernel,
    RowWiseTorchFP8ScaledMMLinearKernel,
)
from vllm.model_executor.layers.fusion.quant_activation import (
    QuantizedActivation,
    get_input_quant_key,
)
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.online.base import (
    OnlineQuantizationConfig,
)
from vllm.model_executor.layers.quantization.online.fp8 import (
    Fp8PtpcOnlineLinearMethod,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8DynamicTokenSym,
)
from vllm.platforms import current_platform

# Quick all-reduce levels whose error on top of the FP8 GEMMs costs too much accuracy.
_UNSUPPORTED_QUICK_REDUCE = ("INT4", "INT3")
# Kernels that keep the `[N, K]` FP8 weight bytes the Ulysses prefill gathers.
_PLAIN_LAYOUT_KERNELS = (
    AiterPerTokenFp8ScaledMMLinearKernel,
    RowWiseTorchFP8ScaledMMLinearKernel,
)


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


def use_online_fp8(layer: LinearBase, quant_config: QuantizationConfig | None) -> bool:
    """Check an online `fp8_per_channel` layer for the Muse FP8 paths.

    Returns:
        Whether the layer is FP8; False if no online target applies to it.

    Raises:
        NotImplementedError: For another online target (including a Hadamard
            wrapper or weight-only FP8), a requantized checkpoint layer, or off
            gfx950 / without AITER.
        ValueError: With LoRA (its wrappers do not take the fused norms' FP8
            input), `--monokernel` (it reads the BF16 weights), or a coarse quick
            all-reduce (INT4 / INT3), whose quantization error on top of FP8 GEMMs
            costs too much accuracy.

    """
    if not online_target(layer, quant_config):
        return False
    method = layer.quant_method
    if (
        type(method) is not Fp8PtpcOnlineLinearMethod
        or method.requantization_source is not None
        or method.activation_quant_key != kFp8DynamicTokenSym
    ):
        raise NotImplementedError(
            f"Muse Spark supports only the 'fp8_per_channel' online quantization "
            f"of BF16 weights with per-token FP8 activations on {layer.prefix}; got "
            f"{type(method).__name__}."
        )
    from vllm._aiter_ops import is_aiter_found_and_supported

    supported = current_platform.is_rocm()
    if supported:
        from vllm.platforms.rocm import on_gfx950

        supported = on_gfx950() and is_aiter_found_and_supported()
    if not supported:
        raise NotImplementedError(
            "Muse Spark FP8 dense linears need ROCm gfx950 with AITER."
        )
    vllm_config = get_current_vllm_config()
    if vllm_config.lora_config is not None:
        raise ValueError(
            "Muse Spark FP8 dense linears do not support LoRA: the fused norms feed "
            "the base layers pre-quantized FP8 input."
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
    return True


def is_fp8(layer: torch.nn.Module | None) -> bool:
    return isinstance(getattr(layer, "quant_method", None), Fp8PtpcOnlineLinearMethod)


def takes_fp8_input(layer: torch.nn.Module | None) -> bool:
    """Whether `layer`'s kernel takes per-token FP8 rows (`fp8_input`)."""
    return layer is not None and get_input_quant_key(layer) == kFp8DynamicTokenSym


def fp8_input(
    xq: torch.Tensor, xs: torch.Tensor, dtype: torch.dtype
) -> QuantizedActivation:
    """Per-token FP8 rows `xq` and their scales `xs` as a linear layer's input."""
    return QuantizedActivation(
        data=xq,
        scale=xs.view(-1, 1),
        orig_dtype=dtype,
        orig_shape=xq.shape,
        quant_key=kFp8DynamicTokenSym,
    )


def require_plain_layout(layer: torch.nn.Module) -> None:
    """Refuse an FP8 kernel that reorders the weight bytes the Ulysses prefill
    gathers."""
    kernel = getattr(layer.quant_method, "fp8_linear", None)
    if not isinstance(kernel, _PLAIN_LAYOUT_KERNELS):
        raise NotImplementedError(
            f"Muse Spark ulysses_prefill gathers the FP8 {layer.prefix} weight in "
            f"[N, K] layout, which {type(kernel).__name__} does not keep."
        )


def alias_weight_rows(layer: torch.nn.Module, full: torch.Tensor, rank: int) -> None:
    """Make the layer's `[N, K]` FP8 weight the view of rows `[rank*N, (rank+1)*N)`
    of `full`, the rank-major concatenation of all ranks' weights, keeping its
    layout (a `[K, N]` view for RowWise)."""
    w = layer.weight.data
    wq = w if w.is_contiguous() else w.t()
    rows = wq.shape[0]
    own = full[rank * rows : (rank + 1) * rows]
    if (
        own.shape != wq.shape
        or own.dtype != wq.dtype
        or not torch.equal(own.view(torch.uint8), wq.view(torch.uint8))
    ):
        raise RuntimeError(
            f"Rows {rank * rows}:{(rank + 1) * rows} of the fused FP8 weight are not "
            f"the {tuple(wq.shape)} {wq.dtype} weight of "
            f"{getattr(layer, 'prefix', type(layer).__name__)}."
        )
    layer.weight.data = own if w.is_contiguous() else own.t()


def full_weight_parts(layer: torch.nn.Module, group, world: int):
    """All-gather the `[N, K]` FP8 shards and their row scales: `([wq_r], [ws_r])`
    in rank order.
    """
    w = layer.weight.data
    # RowWise keeps the `[K, N]` view of the `[N, K]` tensor; AiterPerToken stores
    # the `[N, K]` tensor itself.
    wq = w if w.is_contiguous() else w.t()
    assert wq.is_contiguous()
    ws = layer.weight_scale.data.view(-1)
    wparts = [torch.empty_like(wq.view(torch.uint8)) for _ in range(world)]
    torch.distributed.all_gather(wparts, wq.view(torch.uint8), group=group)
    sparts = [torch.empty_like(ws) for _ in range(world)]
    torch.distributed.all_gather(sparts, ws, group=group)
    return [p.view(wq.dtype) for p in wparts], sparts

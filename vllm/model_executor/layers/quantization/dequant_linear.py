# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Linear layers that a checkpoint stores in FP8 and the model runs in BF16."""

from collections.abc import Callable
from types import MethodType

import torch

from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.utils import (
    _in_own_storage,
    is_weights_pre_processed,
    retire_parameter,
    storage_view,
)


class DequantLinearMethod(LinearMethodBase):
    """A layer with an FP8 checkpoint weight that runs in BF16.

    `base`, the checkpoint's linear method, created the checkpoint's
    parameters, so `weight` and its scale load independently (in any order
    and any number of `load_weights` calls). The weight may also arrive whole
    in BF16, as RL trainers send it; it then needs no scale. Post-load
    processing dequantizes an FP8 weight (per-tensor, per-channel, block or MX
    scales) to BF16 and retires the scales. A reload first restores the
    checkpoint layout (direct reload views the BF16 storage as FP8, layerwise
    reload restages the layer), so each update dequantizes what it loaded."""

    supports_pre_processed_weights = True

    def __init__(self, base: QuantizeMethodBase):
        self.base = base
        self.bf16_method = UnquantizedLinearMethod()

    @classmethod
    def wrap(cls, layer: LinearBase) -> bool:
        """Run a built layer with an FP8 checkpoint weight in BF16. Returns
        whether the layer runs this way."""
        if isinstance(layer.quant_method, cls):
            return True
        weight = getattr(layer, "weight", None)  # None: e.g. packed FP4
        if (
            weight is None
            or weight.dtype != torch.float8_e4m3fn
            # online quantization: an FP8 weight loaded from BF16 tensors
            or getattr(layer, "quantizes_on_load", False)
        ):
            return False
        layer.quant_method = cls(layer.quant_method)
        _load_weight_in_sent_dtype(layer)
        return True

    def create_weights(self, layer: torch.nn.Module, *args, **kwargs) -> None:
        self.base.create_weights(layer, *args, **kwargs)
        _load_weight_in_sent_dtype(layer)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if layer.weight.dtype == torch.float8_e4m3fn and not is_weights_pre_processed():
            # into the loaded storage (allocated BF16-sized): one allocation
            layer.weight.data = _in_own_storage(
                layer.weight, _dequantize(layer), whole_storage=True
            )
        for name, _ in layer.named_parameters(recurse=False):
            if name not in ("weight", "bias"):
                retire_parameter(layer, name)
        self.bf16_method.process_weights_after_loading(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.bf16_method.apply(layer, x, bias)


def bf16_linear(
    build: Callable[..., LinearBase], quant_config: QuantizationConfig | None
) -> LinearBase:
    """`build(quant_config=...)`, a linear layer that runs in BF16: with the
    checkpoint's FP8 weight (`DequantLinearMethod.wrap`), else unquantized.
    Any other quantized layer (e.g. online quantization) is rebuilt
    unquantized."""
    layer = build(quant_config=quant_config)
    if DequantLinearMethod.wrap(layer) or isinstance(
        layer.quant_method, UnquantizedLinearMethod
    ):
        return layer
    return build(quant_config=None)


def _dequantize(layer: torch.nn.Module) -> torch.Tensor:
    """The FP8 weight times its scale, in the layer's dtype. A block scale
    covers `weight_block_size` (the last blocks may be partial; ModelOpt
    registers it 4-D as `[out_blocks, 1, in_blocks, 1]`); a 1-D scale covers a
    channel or a merged shard (per-tensor)."""
    weight = layer.weight
    rows, cols = weight.shape
    scale = getattr(layer, "weight_scale_inv", None)
    if scale is None:
        scale = layer.weight_scale
    if scale.dtype == torch.uint8:  # MX: E8M0 exponents
        scale = scale.view(torch.float8_e8m0fnu)
    scale = scale.to(torch.float32)
    if scale.ndim == 4:
        scale = scale.squeeze(3).squeeze(1)
    if scale.ndim < 2:
        if scale.numel() != rows:  # per tensor: one scale per merged shard
            widths = torch.tensor(layer.logical_widths, device=scale.device)
            scale = scale.reshape(-1).repeat_interleave(widths)
        scale = scale.reshape(rows, 1)
    block_n, block_k = getattr(layer, "weight_block_size", None) or (
        rows // scale.shape[0],
        cols // scale.shape[1],
    )
    scale = scale.repeat_interleave(block_n, 0)[:rows]
    scale = scale.repeat_interleave(block_k, 1)[:, :cols]
    return weight.to(torch.float32).mul_(scale).to(layer.params_dtype)


def _view_as(param: torch.Tensor, dtype: torch.dtype) -> torch.Tensor | None:
    """`param`'s storage viewed in `dtype` with its shape, if it fits."""
    storage = param.untyped_storage()
    if (
        param.is_contiguous()
        and param.storage_offset() == 0
        and storage.nbytes() >= param.numel() * dtype.itemsize
    ):
        return storage_view(storage, 0, param.shape, param.stride(), dtype)
    return None


def _load_weight_in_sent_dtype(layer: torch.nn.Module) -> None:
    """Keep a whole weight in the dtype it is sent in: FP8, or BF16 (no scale
    needed), on storage sized for BF16. The loader is bound to the layer, so
    layerwise reload's metadata does not keep the layer alive
    (`sanitize_layer_refs`)."""
    param = layer.weight
    bf16 = torch.empty(param.shape, dtype=layer.params_dtype, device=param.device)
    param.data = storage_view(
        bf16.untyped_storage(), 0, param.shape, param.stride(), param.dtype
    )
    layer.checkpoint_weight_loader = param.weight_loader
    param.weight_loader = MethodType(_weight_loader_in_sent_dtype, layer)


def _weight_loader_in_sent_dtype(
    layer: torch.nn.Module,
    param: torch.Tensor,
    loaded_weight: torch.Tensor,
    *args,
    **kwargs,
):
    """View the parameter in the sent dtype (on its storage when it fits)."""
    fp8 = loaded_weight.dtype == torch.float8_e4m3fn
    dtype = torch.float8_e4m3fn if fp8 else layer.params_dtype
    if param.dtype != dtype:
        view = _view_as(param, dtype)
        param.data = torch.empty_like(param, dtype=dtype) if view is None else view
    return layer.checkpoint_weight_loader(param, loaded_weight, *args, **kwargs)

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Block-FP8 checkpoint weights for linear layers that run in BF16.

Some models keep a projection in BF16 although their FP8 checkpoint stores it
as block-FP8 (``weight`` + ``weight_scale_inv``). The weight and its scale are
registered under their checkpoint names and loaded independently, in any order
and in any number of ``load_weights`` calls; post-load processing dequantizes
them. A BF16 tensor under the weight's name (a BF16 checkpoint, or a trainer
sending BF16 weights) is loaded as is and needs no scale.

Each FP8 row sits in the first half of the bytes of its BF16 row, so the
weight's storage is the size of the BF16 weight and every row can be loaded as
either format, even within one fused weight.
"""

from typing import Any

import torch

from vllm.model_executor.layers.linear import (
    LinearBase,
    UnquantizedLinearMethod,
    register_weight_loader_v2_supported_method,
)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    create_fp8_scale_parameter,
    validate_fp8_block_shape,
)
from vllm.model_executor.parameter import (
    BlockQuantScaleParameter,
    ModelWeightParameter,
)
from vllm.model_executor.utils import (
    is_weights_pre_processed,
    retire_parameter,
    set_weight_attrs,
)
from vllm.utils.math_utils import cdiv

# Rows dequantized at once; bounds the float32 temporaries of a large weight.
_DEQUANT_ROWS = 1024


def _fp8_view_of(runtime: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """The FP8 checkpoint layout in ``runtime``'s storage: row i in the first
    half of the bytes of runtime row i."""
    view = torch.empty(0, dtype=torch.float8_e4m3fn, device=runtime.device)
    offset = runtime.storage_offset() * runtime.element_size()
    return view.set_(runtime.untyped_storage(), offset, (rows, cols), (2 * cols, 1))


def _set_data(param: torch.nn.Parameter, tensor: torch.Tensor) -> None:
    param.data = tensor


class Fp8DequantWeightParameter(ModelWeightParameter):
    """A ``[rows, cols]`` BF16 weight loadable from block-FP8 or BF16 rows.

    ``data`` is the FP8 view (row stride ``2 * cols`` bytes) while loading and
    the BF16 view after post-load processing; both address the same rows of one
    storage. ``fp8_rows`` marks the rows whose bytes are FP8 and still need
    dequantizing."""

    runtime_dtype: torch.dtype
    fp8_rows: torch.Tensor

    @classmethod
    def empty(
        cls, rows: int, cols: int, dtype: torch.dtype, **kwargs
    ) -> "Fp8DequantWeightParameter":
        """An uninitialized ``[rows, cols]`` weight running in ``dtype``."""
        assert dtype.itemsize == 2, dtype
        runtime = torch.empty(rows, cols, dtype=dtype)
        param = cls(data=_fp8_view_of(runtime, rows, cols), **kwargs)
        param.runtime_dtype = dtype
        param.fp8_rows = torch.zeros(rows, dtype=torch.bool)
        return param

    def fp8_view(self) -> torch.Tensor:
        """The weight's storage in the checkpoint's FP8 layout. Derived from
        the runtime layout, never from ``data``'s strides: a copy of ``data``
        may have lost them (moving to the meta device makes it contiguous)."""
        return _fp8_view_of(self.runtime_view(), *self.data.shape)

    def runtime_view(self) -> torch.Tensor:
        """The weight's storage as the contiguous runtime-dtype weight."""
        data = self.data
        rows, cols = data.shape
        storage = data.untyped_storage()
        offset = data.storage_offset() * data.element_size()
        if storage.nbytes() < offset + 2 * rows * cols:
            # the storage only spans the FP8 rows (e.g. materialized from the
            # FP8 layout for a layerwise reload). Grown in place: a copy would
            # count as loaded elements in the layerwise reload's accounting.
            storage.resize_(offset + 2 * rows * cols)
        view = torch.empty(0, dtype=self.runtime_dtype, device=data.device)
        return view.set_(storage, offset // 2, (rows, cols), (cols, 1))

    def _load(self, load, loaded_weight: torch.Tensor, rows: slice, **kwargs):
        fp8 = loaded_weight.dtype == torch.float8_e4m3fn
        _set_data(self, self.fp8_view())
        if not fp8:
            _set_data(self, self.runtime_view())
        try:
            load(loaded_weight, **kwargs)
        finally:
            _set_data(self, self.fp8_view())
        self.fp8_rows[rows] = fp8

    def load_column_parallel_weight(self, loaded_weight: torch.Tensor):
        self._load(super().load_column_parallel_weight, loaded_weight, slice(None))

    def load_row_parallel_weight(self, loaded_weight: torch.Tensor):
        self._load(super().load_row_parallel_weight, loaded_weight, slice(None))

    def load_merged_column_weight(self, loaded_weight: torch.Tensor, **kwargs):
        start = kwargs["shard_offset"]
        rows = slice(start, start + kwargs["shard_size"])
        self._load(super().load_merged_column_weight, loaded_weight, rows, **kwargs)

    def load_qkv_weight(self, loaded_weight: torch.Tensor, **kwargs):
        start = kwargs["shard_offset"]
        rows = slice(start, start + kwargs["shard_size"])
        self._load(super().load_qkv_weight, loaded_weight, rows, **kwargs)

    @torch.no_grad()
    def dequantize(self, scale_inv: torch.Tensor, block_size: list[int]) -> None:
        """Dequantize the FP8 rows in place and switch ``data`` to the runtime
        view. Every row then holds runtime bytes, so a later reload that leaves
        this weight out keeps them."""
        runtime = self.runtime_view()
        fp8 = self.fp8_view()  # after runtime_view, which may move the storage
        block_n, block_k = block_size
        cols = fp8.shape[1]
        chunk = max(block_n, _DEQUANT_ROWS // block_n * block_n)
        flags = self.fp8_rows.tolist()
        start = 0
        while start < len(flags):
            if not flags[start]:
                start += 1
                continue
            end = start
            while end < len(flags) and flags[end] and end - start < chunk:
                end += 1
            if start % block_n:
                raise ValueError(
                    f"FP8 rows starting at {start} are not aligned to the "
                    f"{block_n}-row scale blocks"
                )
            scale = scale_inv[start // block_n : cdiv(end, block_n)].to(torch.float32)
            scale = scale.repeat_interleave(block_n, dim=0)[: end - start]
            scale = scale.repeat_interleave(block_k, dim=1)[:, :cols]
            # computed before writing: BF16 row i overwrites FP8 row i
            out = (fp8[start:end].to(torch.float32) * scale).to(self.runtime_dtype)
            runtime[start:end].copy_(out)
            start = end
        self.fp8_rows.zero_()
        _set_data(self, runtime)


@register_weight_loader_v2_supported_method
class Fp8DequantLinearMethod(UnquantizedLinearMethod):
    """Loads block-FP8 (or BF16) checkpoint weights; runs in BF16."""

    def __init__(self, weight_block_size: list[int]):
        super().__init__()
        self.weight_block_size = weight_block_size

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        weight_loader = extra_weight_attrs.pop("weight_loader")
        layer.weight_block_size = self.weight_block_size
        validate_fp8_block_shape(
            layer,
            input_size,
            output_size,
            input_size_per_partition,
            output_partition_sizes,
            self.weight_block_size,
        )
        weight = Fp8DequantWeightParameter.empty(
            sum(output_partition_sizes),
            input_size_per_partition,
            params_dtype,
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

        scale = create_fp8_scale_parameter(
            BlockQuantScaleParameter,
            output_partition_sizes,
            input_size_per_partition,
            self.weight_block_size,
            weight_loader,
        )
        scale.weight_block_size = self.weight_block_size
        layer.register_parameter("weight_scale_inv", scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # pre-processed weights (weight cache) are runtime BF16 already
        if not is_weights_pre_processed():
            layer.weight.dequantize(layer.weight_scale_inv, self.weight_block_size)
        retire_parameter(layer, "weight_scale_inv")
        super().process_weights_after_loading(layer)


class Fp8DequantConfig(QuantizationConfig):
    """Quantization config for linear layers that load a block-FP8 checkpoint
    and run in BF16 (see `Fp8DequantLinearMethod`). Models pass it to the
    layers concerned; it is never read from a checkpoint."""

    def __init__(self, weight_block_size: list[int]):
        super().__init__()
        self.weight_block_size = weight_block_size

    @classmethod
    def for_checkpoint(
        cls, quant_config: QuantizationConfig | None
    ) -> "Fp8DequantConfig | None":
        """A config for layers of a quantized checkpoint that the model keeps
        in BF16 but the checkpoint may store as block-FP8 (or keep in BF16);
        None for an unquantized checkpoint."""
        if quant_config is None:
            return None
        block_size = getattr(quant_config, "weight_block_size", None)
        return cls(list(block_size) if block_size else [128, 128])

    def get_name(self) -> QuantizationMethods:
        return "fp8"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 0

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Fp8DequantConfig":
        return cls(config["weight_block_size"])

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        if isinstance(layer, LinearBase):
            return Fp8DequantLinearMethod(self.weight_block_size)
        return None

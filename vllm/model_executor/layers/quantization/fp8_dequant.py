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

A replicated layer may also accept its scale in any layout a checkpoint uses
(per-tensor, per-channel, blocks, MXFP8): the scale is then kept as one value
per row and 32 columns, the finest of these layouts, and a coarser checkpoint
scale is expanded to it when loaded.
"""

import weakref
from typing import Any

import torch

from vllm.model_executor.layers.linear import (
    LinearBase,
    ReplicatedLinear,
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
# Scale blocks of a weight that accepts any checkpoint scale layout.
_ANY_LAYOUT_BLOCK = [1, 32]


def _fp8_view_of(runtime: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """The FP8 checkpoint layout in ``runtime``'s storage: row i in the first
    half of the bytes of runtime row i."""
    view = torch.empty(0, dtype=torch.float8_e4m3fn, device=runtime.device)
    offset = runtime.storage_offset() * runtime.element_size()
    return view.set_(runtime.untyped_storage(), offset, (rows, cols), (2 * cols, 1))


def _set_data(param: torch.nn.Parameter, tensor: torch.Tensor) -> None:
    # assigning `self.data` inside the subclass trips mypy's [has-type]
    param.data = tensor


class Fp8DequantWeightParameter(ModelWeightParameter):
    """A ``[rows, cols]`` BF16 weight loadable from block-FP8 or BF16 rows.

    ``data`` is the BF16 weight. An FP8 tensor loads through the FP8 view of
    the same storage (row stride ``2 * cols`` bytes, row i in the first half of
    BF16 row i's bytes); ``fp8_rows`` marks the rows whose bytes are FP8 and
    still need dequantizing. Rows loaded as BF16, or never loaded (dummy
    weights), hold runtime bytes."""

    runtime_dtype: torch.dtype
    fp8_rows: torch.Tensor
    layer_ref: "weakref.ref[torch.nn.Module]"

    @classmethod
    def empty(
        cls, rows: int, cols: int, dtype: torch.dtype, **kwargs
    ) -> "Fp8DequantWeightParameter":
        """An uninitialized ``[rows, cols]`` weight running in ``dtype``."""
        assert dtype.itemsize == 2, dtype
        param = cls(data=torch.empty(rows, cols, dtype=dtype), **kwargs)
        param.runtime_dtype = dtype
        # host bookkeeping: also when the model is built on meta (weight cache)
        param.fp8_rows = torch.zeros(rows, dtype=torch.bool, device="cpu")
        return param

    def fp8_view(self) -> torch.Tensor:
        """The weight's storage in the checkpoint's FP8 layout."""
        return _fp8_view_of(self.runtime_view(), *self.data.shape)

    def runtime_view(self) -> torch.Tensor:
        """The weight's storage as the contiguous runtime-dtype weight."""
        data = self.data
        if data.dtype == self.runtime_dtype:
            return data
        offset = data.storage_offset() * data.element_size()  # the FP8 view
        view = torch.empty(0, dtype=self.runtime_dtype, device=data.device)
        return view.set_(
            data.untyped_storage(), offset // 2, data.shape, (data.shape[1], 1)
        )

    def _load(self, load, loaded_weight: torch.Tensor, rows: slice, **kwargs):
        fp8 = loaded_weight.dtype == torch.float8_e4m3fn
        if fp8:
            runtime = self.data
            _set_data(self, self.fp8_view())
            try:
                load(loaded_weight, **kwargs)
            finally:
                _set_data(self, runtime)
        else:
            load(loaded_weight, **kwargs)
        self.fp8_rows[rows] = fp8

    def load_column_parallel_weight(self, loaded_weight: torch.Tensor):
        self._load(super().load_column_parallel_weight, loaded_weight, slice(None))

    def load_replicated_weight(self, loaded_weight: torch.Tensor):
        self._load(self._assert_and_load, loaded_weight, slice(None))

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
        fp8 = self.fp8_view()
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


def _scale_groups(scale: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """The checkpoint scale of a ``[rows, cols]`` FP8 weight, in any layout
    (per-tensor, per-channel, ``[n, k]`` blocks, MXFP8 E8M0), as one float32
    scale per row and ``_ANY_LAYOUT_BLOCK[1]`` columns."""
    group = _ANY_LAYOUT_BLOCK[1]
    if scale.dtype == torch.float32 and tuple(scale.shape) == (rows, cdiv(cols, group)):
        return scale  # in this layout already (e.g. `load_unneeded_scale`)
    if scale.dtype in (torch.uint8, torch.float8_e8m0fnu):
        scale = scale.view(torch.float8_e8m0fnu)
    scale = scale.to(torch.float32)
    scale = scale.reshape(scale.shape[0] if scale.ndim else 1, -1)
    block_n, block_k = rows // scale.shape[0], cols // scale.shape[1]
    if (
        block_n * scale.shape[0] != rows
        or block_k * scale.shape[1] != cols
        or (block_k % group and scale.shape[1] > 1)
    ):
        raise ValueError(
            f"unsupported scale shape {tuple(scale.shape)} for a "
            f"[{rows}, {cols}] FP8 weight"
        )
    columns = torch.arange(0, cols, group, device=scale.device) // block_k
    return scale.repeat_interleave(block_n, dim=0)[:, columns]


class Fp8DequantScaleParameter(BlockQuantScaleParameter):
    """The scale of a replicated weight that accepts any checkpoint scale
    layout, stored per ``_ANY_LAYOUT_BLOCK`` (see `_scale_groups`). Only for
    replicated (``disable_tp``) layers: a checkpoint scale is expanded before
    it is sliced, so it must hold the whole weight's rows."""

    cols: int

    def load_column_parallel_weight(self, loaded_weight: torch.Tensor):
        loaded_weight = _scale_groups(loaded_weight, self.data.shape[0], self.cols)
        super().load_column_parallel_weight(loaded_weight)

    def load_replicated_weight(self, loaded_weight: torch.Tensor):
        loaded_weight = _scale_groups(loaded_weight, self.data.shape[0], self.cols)
        self._assert_and_load(loaded_weight)

    def load_merged_column_weight(self, loaded_weight: torch.Tensor, **kwargs):
        rows = kwargs["shard_size"]  # weight rows: one scale row per row
        loaded_weight = _scale_groups(loaded_weight, rows, self.cols)
        super().load_merged_column_weight(loaded_weight, **kwargs)


def _load_replicated(param: torch.nn.Parameter, loaded_weight: torch.Tensor):
    """weight_loader of a ReplicatedLinear's parameters: its own loader copies
    into ``data`` and would bypass the FP8 view (and the scale expansion)."""
    if hasattr(param, "load_replicated_weight"):
        param.load_replicated_weight(loaded_weight)
    else:
        param._assert_and_load(loaded_weight)


def load_unneeded_scale(
    name: str,
    param: torch.nn.Parameter,
    tensor: torch.Tensor,
    loaded: set[str],
    shard_id: int | str | None = None,
) -> None:
    """After ``tensor`` loaded into ``param`` (``name``): a BF16 tensor loaded
    into a block-FP8 checkpoint weight needs no scale. Load ones into the scale
    rows it owns, so that every element of the layer counts as loaded (a
    layerwise reload processes a layer once all of them have arrived) and the
    default loader's load check passes."""
    if (
        not isinstance(param, Fp8DequantWeightParameter)
        or tensor.dtype == torch.float8_e4m3fn
    ):
        return
    layer = param.layer_ref()
    assert layer is not None
    scale = layer.weight_scale_inv
    block_n, block_k = scale.weight_block_size
    shape = (cdiv(tensor.shape[0], block_n), cdiv(tensor.shape[1], block_k))
    ones = tensor.new_ones(shape, dtype=torch.float32)
    scale.weight_loader(scale, ones, *(() if shard_id is None else (shard_id,)))
    loaded.add(name.removesuffix("weight") + "weight_scale_inv")


@register_weight_loader_v2_supported_method
class Fp8DequantLinearMethod(UnquantizedLinearMethod):
    """Loads block-FP8 (or BF16) checkpoint weights; runs in BF16.

    ``weight_block_size`` None: the layer is replicated and accepts the scale
    in any checkpoint layout (see `_scale_groups`).

    Model loaders call `load_unneeded_scale` after loading a tensor into such a
    weight, as a separate loader call: a BF16 tensor then loads ones into the
    scale rows it owns, so every element of the layer is loaded (layerwise
    reload counts each loader call; a copy nested in the weight's own call
    would count against the weight)."""

    def __init__(self, weight_block_size: list[int] | None):
        super().__init__()
        self.any_scale_layout = weight_block_size is None
        self.weight_block_size = weight_block_size or _ANY_LAYOUT_BLOCK

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
        if isinstance(layer, ReplicatedLinear):
            weight_loader = _load_replicated
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
        weight.layer_ref = weakref.ref(layer)  # see `load_unneeded_scale`

        scale = create_fp8_scale_parameter(
            BlockQuantScaleParameter,
            output_partition_sizes,
            input_size_per_partition,
            self.weight_block_size,
            weight_loader,
        )
        if self.any_scale_layout:
            if getattr(layer, "tp_size", 1) > 1:
                raise ValueError("a scale in any layout needs a replicated layer")
            scale = Fp8DequantScaleParameter(
                data=scale.data, input_dim=1, output_dim=0, weight_loader=weight_loader
            )
            scale.cols = input_size_per_partition
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

    def __init__(self, weight_block_size: list[int] | None):
        super().__init__()
        self.weight_block_size = weight_block_size

    @classmethod
    def for_checkpoint(
        cls, quant_config: QuantizationConfig | None, any_scale_layout: bool = False
    ) -> "Fp8DequantConfig | None":
        """A config for layers of a quantized checkpoint that the model keeps
        in BF16 but the checkpoint may store as block-FP8 (or keep in BF16);
        None for an unquantized checkpoint. With ``any_scale_layout``, for
        replicated layers whose FP8 scale layout varies between checkpoints,
        the layers accept a scale in any layout."""
        if quant_config is None:
            return None
        if any_scale_layout:
            return cls(None)
        block_size = getattr(quant_config, "weight_block_size", None)
        return cls(list(block_size) if block_size else [128, 128])

    def get_name(self) -> QuantizationMethods:
        # never the model's quantization config, so nothing selects on it
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
        # built by models (`for_checkpoint`), never read from a checkpoint
        raise NotImplementedError

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        if isinstance(layer, LinearBase):
            return Fp8DequantLinearMethod(self.weight_block_size)
        return None

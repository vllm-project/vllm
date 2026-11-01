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
    MergedColumnParallelLinear,
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
    FP8_SCALE_SENTINEL,
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


def _set_data(param: torch.nn.Parameter, tensor: torch.Tensor) -> None:
    # assigning `self.data` inside the subclass trips mypy's [has-type]
    param.data = tensor


class Fp8DequantWeightParameter(ModelWeightParameter):
    """A ``[rows, cols]`` BF16 weight loadable from block-FP8 or BF16 rows.

    An FP8 tensor loads through `fp8_view`, the same storage in the
    checkpoint's layout; ``fp8_rows`` (on the host) marks the rows whose bytes
    are FP8 and still need dequantizing. Rows loaded as BF16, or never loaded
    (dummy weights), hold runtime bytes."""

    fp8_rows: torch.Tensor
    layer_ref: "weakref.ref[torch.nn.Module]"

    def fp8_view(self) -> torch.Tensor:
        """The weight's storage in the checkpoint's FP8 layout: row i in the
        first half of the bytes of BF16 row i."""
        data = self.data
        rows, cols = data.shape
        view = torch.empty(0, dtype=torch.float8_e4m3fn, device=data.device)
        offset = data.storage_offset() * data.element_size()
        return view.set_(data.untyped_storage(), offset, (rows, cols), (2 * cols, 1))

    def _load(self, load, loaded_weight: torch.Tensor, **kwargs):
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
        rows = slice(None)  # the whole weight, or a merged / qkv shard's rows
        if "shard_offset" in kwargs:
            start = kwargs["shard_offset"]
            rows = slice(start, start + kwargs["shard_size"])
        # loader state for PWAL (RFC #64418 sec. 5 rule 3 exception): load, never copy
        self.fp8_rows[rows] = fp8

    def load_column_parallel_weight(self, loaded_weight: torch.Tensor):
        self._load(super().load_column_parallel_weight, loaded_weight)

    def load_replicated_weight(self, loaded_weight: torch.Tensor):
        self._load(self._assert_and_load, loaded_weight)

    def load_row_parallel_weight(self, loaded_weight: torch.Tensor):
        self._load(super().load_row_parallel_weight, loaded_weight)

    def load_merged_column_weight(self, loaded_weight: torch.Tensor, **kwargs):
        self._load(super().load_merged_column_weight, loaded_weight, **kwargs)

    def load_qkv_weight(self, loaded_weight: torch.Tensor, **kwargs):
        self._load(super().load_qkv_weight, loaded_weight, **kwargs)

    @torch.no_grad()
    def dequantize(self, scale_inv: torch.Tensor, block_size: list[int]) -> None:
        """Dequantize the FP8 rows in place. Every row then holds runtime
        bytes, so a later reload that leaves this weight out keeps them."""
        if not self.fp8_rows.any():  # host flags: no device sync
            return
        runtime = self.data
        fp8 = self.fp8_view().view(torch.uint8)
        block_n, block_k = block_size
        rows = self.fp8_rows.nonzero().flatten().to(runtime.device)
        blocks = rows // block_n
        if (scale_inv[blocks] == FP8_SCALE_SENTINEL).any():
            raise ValueError(
                "FP8 weight rows were loaded without their weight_scale_inv"
            )
        for index in rows.split(_DEQUANT_ROWS):
            # gathered before writing: BF16 row i overwrites FP8 row i
            weight = fp8[index].view(torch.float8_e4m3fn).to(torch.float32)
            scale = scale_inv[index // block_n].to(torch.float32)
            scale = scale.repeat_interleave(block_k, dim=1)[:, : weight.shape[1]]
            runtime[index] = (weight * scale).to(runtime.dtype)
        self.fp8_rows.zero_()


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
    """The scale of a weight that accepts any checkpoint scale layout, stored
    per ``_ANY_LAYOUT_BLOCK`` (see `_scale_groups`). Only for a replicated
    (``disable_tp``) MergedColumnParallelLinear: a checkpoint scale is
    expanded before it is sliced, so it must hold the whole shard's rows."""

    cols: int

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
    """Model loaders call this after loading ``tensor`` into ``param``
    (``name``): a BF16 tensor loaded into a block-FP8 checkpoint weight needs no
    scale, so ones are loaded into the scale rows it owns and every element of
    the layer counts as loaded (layerwise reload processes a layer once all of
    them have arrived, counting each loader call; hence a separate call, not
    one nested in the weight's own)."""
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
    """Loads block-FP8 (or BF16) checkpoint weights; runs in BF16. Model
    loaders call `load_unneeded_scale` after each tensor they load into such a
    weight.

    ``weight_block_size`` None: the layer is replicated and accepts the scale
    in any checkpoint layout (see `_scale_groups`)."""

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
        assert params_dtype.itemsize == 2, params_dtype
        rows, cols = sum(output_partition_sizes), input_size_per_partition
        weight = Fp8DequantWeightParameter(
            data=torch.empty(rows, cols, dtype=params_dtype),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        # host bookkeeping: also when the model is built on meta (weight cache)
        weight.fp8_rows = torch.zeros(rows, dtype=torch.bool, device="cpu")
        weight.layer_ref = weakref.ref(layer)  # see `load_unneeded_scale`
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

        scale = create_fp8_scale_parameter(
            BlockQuantScaleParameter,
            output_partition_sizes,
            input_size_per_partition,
            self.weight_block_size,
            weight_loader,
        )
        # until loaded: `dequantize` rejects FP8 rows whose scale is missing
        scale.data.fill_(FP8_SCALE_SENTINEL)
        if self.any_scale_layout:
            # the sparse-attention indexer's fused wk_weights_proj
            assert isinstance(layer, MergedColumnParallelLinear) and layer.tp_size == 1
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
            # in place, for the next direct reload: FP8 rows it sends without
            # their scale are rejected, not dequantized with this one
            layer.weight_scale_inv.data.fill_(FP8_SCALE_SENTINEL)
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

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""TP full-weight switching for compatible linear quantization methods.

A linear method opts in by inheriting :class:`TPWeightSwitchMixin` and
declaring one :class:`TPWeightGatherSpec` per post-load tensor consumed by its
kernel. The mixin owns the buffers, asynchronous collectives, and local/full
tensor aliases. Tensors whose post-load layout is not directly concatenative on
the sharded axis attach a reusable :class:`TPWeightGatherLayout` to their spec
instead of overriding mixin internals.
"""

from collections.abc import Callable, MutableMapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

import torch

from vllm.utils.math_utils import round_up


def _noop() -> None:
    return None


@dataclass(frozen=True)
class TPWeightGatherSpec:
    """A direct layer tensor attribute to all-gather across TP-like ranks.

    Args:
        attr_name: Layer attribute holding the post-load tensor.
        gather_dim: Axis sharded across the gather group.
        make_full_contiguous: Assemble the gathered tensor into a contiguous
            buffer instead of exposing a moved-dim view.
        layout: Optional adapter for post-load layouts that are not directly
            concatenative on ``gather_dim``.

    """

    attr_name: str
    gather_dim: int = 0
    make_full_contiguous: bool = False
    layout: "TPWeightGatherLayout | None" = None


@dataclass
class TPWeightGatherPart:
    """Runtime tensors for one gathered attribute.

    ``prepare`` refreshes ``gather_input`` from ``tp_tensor`` before every
    collective; ``finalize`` assembles ``full_tensor`` after it completes.
    """

    spec: TPWeightGatherSpec
    tp_tensor: torch.Tensor
    gather_input: torch.Tensor
    gather_output: torch.Tensor
    full_tensor: torch.Tensor
    prepare: Callable[[], None] = _noop
    finalize: Callable[[], None] = _noop


@dataclass
class TPWeightSwitchState:
    """Reusable buffers and outstanding collectives for one linear layer."""

    gather_parts: dict[str, TPWeightGatherPart] = field(default_factory=dict)
    handles: list[Any] = field(default_factory=list)


@dataclass
class TPWeightBufferPool:
    """Allocate full-weight buffers, optionally shared across layers."""

    pool: MutableMapping[Any, torch.Tensor] | None = None
    key_prefix: Any = None

    def get(
        self,
        name: str,
        shape: tuple[int, ...] | list[int],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        key = (
            self.key_prefix,
            name,
            device.type,
            device.index,
            dtype,
            tuple(shape),
        )
        buffer = None if self.pool is None else self.pool.get(key)
        if buffer is None:
            buffer = torch.empty(shape, dtype=dtype, device=device)
            if self.pool is not None:
                self.pool[key] = buffer
        return buffer


class TPWeightGatherLayout:
    """Describe how one post-load tensor is gathered and reassembled."""

    def create_gather_part(
        self,
        layer: torch.nn.Module,
        spec: TPWeightGatherSpec,
        tensor: torch.Tensor,
        tp_size: int,
        buffers: TPWeightBufferPool,
    ) -> TPWeightGatherPart:
        raise NotImplementedError


class ConcatTPWeightLayout(TPWeightGatherLayout):
    """Tensor whose full layout is the concatenation of shards on one axis."""

    def create_gather_part(
        self,
        layer: torch.nn.Module,
        spec: TPWeightGatherSpec,
        tensor: torch.Tensor,
        tp_size: int,
        buffers: TPWeightBufferPool,
    ) -> TPWeightGatherPart:
        del layer
        dim = (
            spec.gather_dim if spec.gather_dim >= 0 else tensor.dim() + spec.gather_dim
        )
        if dim < 0 or dim >= tensor.dim():
            raise RuntimeError(
                f"Invalid TP gather dim {spec.gather_dim} for attribute "
                f"{spec.attr_name!r} with shape {tuple(tensor.shape)}."
            )

        tp_tensor = tensor.detach()
        full_shape = list(tp_tensor.shape)
        full_shape[dim] *= tp_size
        gather_shape = (full_shape[dim], *full_shape[:dim], *full_shape[dim + 1 :])
        gather_input = (
            tp_tensor
            if dim == 0 and tp_tensor.is_contiguous()
            else torch.movedim(tp_tensor, dim, 0).contiguous()
        )
        gather_output = buffers.get(
            f"{spec.attr_name}.gather.dim{dim}",
            gather_shape,
            dtype=tp_tensor.dtype,
            device=tp_tensor.device,
        )
        if dim == 0:
            full_tensor = gather_output
        elif not spec.make_full_contiguous:
            full_tensor = torch.movedim(gather_output, 0, dim)
        else:
            full_tensor = buffers.get(
                f"{spec.attr_name}.full.dim{dim}",
                full_shape,
                dtype=tp_tensor.dtype,
                device=tp_tensor.device,
            )

        def prepare() -> None:
            if gather_input.data_ptr() != tp_tensor.data_ptr():
                gather_input.copy_(torch.movedim(tp_tensor, dim, 0))

        def finalize() -> None:
            if full_tensor.data_ptr() != gather_output.data_ptr():
                full_tensor.copy_(torch.movedim(gather_output, 0, dim))

        return TPWeightGatherPart(
            spec=spec,
            tp_tensor=tp_tensor,
            gather_input=gather_input,
            gather_output=gather_output,
            full_tensor=full_tensor,
            prepare=prepare,
            finalize=finalize,
        )


class NvFp4SwizzledBlockScaleLayout(TPWeightGatherLayout):
    """NVFP4 block scales stored in the CUTLASS 128x4 swizzled layout.

    Swizzled shards are not concatenative on the input axis. The local scale is
    unswizzled into a logical ``[K / group_size, N]`` view, gathered through a
    ``uint8`` alias to preserve FP8 bits, and reswizzled into the full layout
    after the collective completes.
    """

    row_tile: ClassVar[int] = 128
    col_tile: ClassVar[int] = 4

    def __init__(self, group_size: int = 16) -> None:
        self.group_size = group_size

    def create_gather_part(
        self,
        layer: torch.nn.Module,
        spec: TPWeightGatherSpec,
        tensor: torch.Tensor,
        tp_size: int,
        buffers: TPWeightBufferPool,
    ) -> TPWeightGatherPart:
        scale = tensor.detach()
        output_size = layer.output_size_per_partition
        local_scale_cols = layer.input_size_per_partition // self.group_size
        padded_output_size = round_up(output_size, self.row_tile)
        padded_local_scale_cols = round_up(local_scale_cols, self.col_tile)
        expected_shape = (padded_output_size, padded_local_scale_cols)
        if scale.dtype != torch.float8_e4m3fn or tuple(scale.shape) != expected_shape:
            raise RuntimeError(
                "Unexpected CUTLASS-compatible NVFP4 block-scale layout: "
                f"expected dtype=torch.float8_e4m3fn and shape {expected_shape}, "
                f"got dtype={scale.dtype} and shape={tuple(scale.shape)}."
            )

        full_scale_cols = local_scale_cols * tp_size
        padded_full_scale_cols = round_up(full_scale_cols, self.col_tile)
        name = f"{spec.attr_name}.nvfp4_swizzled"
        gather_input = buffers.get(
            f"{name}.gather_input",
            (local_scale_cols, output_size),
            dtype=torch.uint8,
            device=scale.device,
        )
        gather_output = buffers.get(
            f"{name}.gather_output",
            (full_scale_cols, output_size),
            dtype=torch.uint8,
            device=scale.device,
        )
        local_padded = buffers.get(
            f"{name}.local_padded",
            (1, padded_output_size, padded_local_scale_cols),
            dtype=scale.dtype,
            device=scale.device,
        )
        full_padded = buffers.get(
            f"{name}.full_padded",
            (1, padded_output_size, padded_full_scale_cols),
            dtype=scale.dtype,
            device=scale.device,
        )
        full_tensor = buffers.get(
            f"{name}.full",
            (padded_output_size, padded_full_scale_cols),
            dtype=scale.dtype,
            device=scale.device,
        )

        m_tiles = padded_output_size // self.row_tile
        local_k_tiles = padded_local_scale_cols // self.col_tile
        full_k_tiles = padded_full_scale_cols // self.col_tile

        def prepare() -> None:
            local_padded.view(1, m_tiles, 4, 32, local_k_tiles, 4).copy_(
                scale.view(1, m_tiles, local_k_tiles, 32, 4, 4).permute(
                    0, 1, 4, 3, 2, 5
                )
            )
            gather_input.copy_(
                local_padded[0, :output_size, :local_scale_cols].view(torch.uint8).T
            )

        def finalize() -> None:
            full_padded.zero_()
            full_padded[0, :output_size, :full_scale_cols].view(torch.uint8).copy_(
                gather_output.T
            )
            full_tensor.view(1, m_tiles, full_k_tiles, 32, 4, 4).copy_(
                full_padded.view(1, m_tiles, 4, 32, full_k_tiles, 4).permute(
                    0, 1, 4, 3, 2, 5
                )
            )

        return TPWeightGatherPart(
            spec=spec,
            tp_tensor=scale,
            gather_input=gather_input,
            gather_output=gather_output,
            full_tensor=full_tensor,
            prepare=prepare,
            finalize=finalize,
        )


CONCAT_TP_WEIGHT_LAYOUT = ConcatTPWeightLayout()
NVFP4_SWIZZLED_BLOCK_SCALE_LAYOUT = NvFp4SwizzledBlockScaleLayout()


class TPWeightSwitchMixin:
    """Opt-in TP/full-weight switching for a linear method or scheme.

    Subclasses only declare ``tp_weight_gather_specs``. Methods whose support
    depends on the selected kernel may override
    :meth:`get_tp_weight_switch_specs` to validate the post-load layout, so
    unsupported layouts fail closed without quantization-specific branches in
    the caller.
    """

    tp_weight_gather_specs: ClassVar[tuple[TPWeightGatherSpec, ...]] = ()

    @property
    def supports_tp_weight_switch(self) -> bool:
        return bool(self.tp_weight_gather_specs)

    def get_tp_weight_switch_specs(
        self, layer: torch.nn.Module
    ) -> tuple[TPWeightGatherSpec, ...]:
        """Return specs for the method's post-load layout on ``layer``."""
        del layer
        return self.tp_weight_gather_specs

    def enable_tp_weight_switch(
        self,
        layer: torch.nn.Module,
        tp_size: int,
        *,
        pool: MutableMapping[Any, torch.Tensor] | None = None,
        pool_key_prefix: Any = None,
    ) -> TPWeightSwitchState:
        """Allocate local/full tensor aliases and reusable gather buffers."""
        gather_specs = self.get_tp_weight_switch_specs(layer)
        if not gather_specs:
            raise RuntimeError(
                f"{type(self).__name__} did not declare TP weight switch specs."
            )

        buffers = TPWeightBufferPool(pool, pool_key_prefix)
        state = TPWeightSwitchState()
        for spec in gather_specs:
            tensor = getattr(layer, spec.attr_name, None)
            if not isinstance(tensor, torch.Tensor):
                raise RuntimeError(
                    f"{type(self).__name__} declares TP gather attribute "
                    f"{spec.attr_name!r}, but layer "
                    f"{getattr(layer, 'prefix', layer)} does not have it."
                )
            layout = spec.layout or CONCAT_TP_WEIGHT_LAYOUT
            state.gather_parts[spec.attr_name] = layout.create_gather_part(
                layer, spec, tensor, tp_size, buffers
            )
        return state

    def all_gather_tp_weight(
        self,
        state: TPWeightSwitchState,
        group: Any,
        *,
        async_op: bool = True,
    ) -> None:
        """All-gather every input-sharded tensor in ``state``."""
        if state.handles:
            raise RuntimeError(
                "TP weight all-gather is still pending; wait before launching "
                "another one."
            )
        device_group = getattr(group, "device_group", group)
        try:
            for part in state.gather_parts.values():
                part.prepare()
                handle = torch.distributed.all_gather_into_tensor(
                    part.gather_output,
                    part.gather_input,
                    group=device_group,
                    async_op=async_op,
                )
                if handle is not None:
                    state.handles.append(handle)
        except Exception:
            self.wait_tp_weight_all_gather(state)
            raise

    def wait_tp_weight_all_gather(self, state: TPWeightSwitchState) -> None:
        try:
            for handle in state.handles:
                handle.wait()
            for part in state.gather_parts.values():
                part.finalize()
        finally:
            state.handles.clear()

    @staticmethod
    def switch_tp_weight(
        layer: torch.nn.Module,
        state: TPWeightSwitchState,
        *,
        use_full_weight: bool,
    ) -> None:
        """Switch direct tensor attributes between local and full storage."""
        for attr_name, part in state.gather_parts.items():
            target = part.full_tensor if use_full_weight else part.tp_tensor
            with torch.no_grad():
                getattr(layer, attr_name).set_(target)

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""TP full-weight switching for compatible linear quantization methods."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar

import torch


@dataclass(frozen=True)
class TPWeightGatherSpec:
    """A direct layer tensor attribute to all-gather across TP-like ranks."""

    attr_name: str
    gather_dim: int = 0
    make_full_contiguous: bool = False


@dataclass
class TPWeightGatherPart:
    """Runtime tensors for one gathered attribute."""

    spec: TPWeightGatherSpec
    tp_tensor: torch.Tensor
    gather_source: torch.Tensor
    gather_input: torch.Tensor
    gather_output: torch.Tensor
    full_tensor: torch.Tensor
    prepare: Callable[[], None] | None = None
    finalize: Callable[[], None] | None = None


@dataclass
class TPWeightSwitchState:
    """Reusable buffers and outstanding collectives for one linear layer."""

    gather_parts: dict[str, TPWeightGatherPart] = field(default_factory=dict)
    handles: list[Any] = field(default_factory=list)


class TPWeightSwitchMixin:
    """Opt-in TP/full-weight switching for a linear method or scheme.

    Each method declares the tensors consumed by its post-load kernel layout.
    Keeping the switching mechanics on the method makes unsupported layouts
    fail closed without quantization-specific branches in the PCP layer.
    """

    tp_weight_gather_specs: ClassVar[tuple[TPWeightGatherSpec, ...]] = ()
    supports_tp_weight_switch: ClassVar[bool] = False

    def get_tp_weight_switch_specs(
        self, layer: torch.nn.Module
    ) -> tuple[TPWeightGatherSpec, ...]:
        """Return specs for the method's post-load layout on ``layer``."""
        del layer
        return self.tp_weight_gather_specs

    @staticmethod
    def _get_or_create_tp_weight_buffer(
        pool: Any | None,
        key: Any,
        shape: tuple[int, ...] | list[int],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        buffer = None if pool is None else pool.get(key)
        if buffer is None:
            buffer = torch.empty(shape, dtype=dtype, device=device)
            if pool is not None:
                pool[key] = buffer
        return buffer

    def _create_tp_weight_gather_part(
        self,
        layer: torch.nn.Module,
        spec: TPWeightGatherSpec,
        tp_size: int,
        *,
        pool: Any | None,
        pool_key_prefix: Any | None,
    ) -> TPWeightGatherPart:
        """Build runtime state for one direct, concatenative tensor layout.

        Quantization methods with a post-load layout that is not directly
        concatenative may override this method for the affected attribute. The
        returned ``gather_source`` is refreshed into ``gather_input`` before
        every collective, and ``finalize`` may assemble the kernel's full
        layout after the collective completes.
        """
        tensor = getattr(layer, spec.attr_name, None)
        if not isinstance(tensor, torch.Tensor):
            raise RuntimeError(
                f"{type(self).__name__} declares TP gather attribute "
                f"{spec.attr_name!r}, but layer "
                f"{getattr(layer, 'prefix', layer)} does not have it."
            )
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
        gather_shape = (
            full_shape[dim],
            *full_shape[:dim],
            *full_shape[dim + 1 :],
        )
        gather_input = (
            tp_tensor
            if dim == 0 and tp_tensor.is_contiguous()
            else torch.movedim(tp_tensor, dim, 0).contiguous()
        )
        pool_key = (
            pool_key_prefix,
            spec.attr_name,
            tp_tensor.device.type,
            tp_tensor.device.index,
            tp_tensor.dtype,
            dim,
            spec.make_full_contiguous,
            tuple(full_shape),
        )
        gather_output = self._get_or_create_tp_weight_buffer(
            pool,
            (*pool_key, "gather"),
            gather_shape,
            dtype=tp_tensor.dtype,
            device=tp_tensor.device,
        )
        if dim == 0:
            full_tensor = gather_output
        elif not spec.make_full_contiguous:
            full_tensor = torch.movedim(gather_output, 0, dim)
        else:
            full_tensor = self._get_or_create_tp_weight_buffer(
                pool,
                (*pool_key, "full"),
                full_shape,
                dtype=tp_tensor.dtype,
                device=tp_tensor.device,
            )
        return TPWeightGatherPart(
            spec=spec,
            tp_tensor=tp_tensor,
            gather_source=tp_tensor,
            gather_input=gather_input,
            gather_output=gather_output,
            full_tensor=full_tensor,
        )

    def enable_tp_weight_switch(
        self,
        layer: torch.nn.Module,
        tp_size: int,
        *,
        pool: Any | None = None,
        pool_key_prefix: Any | None = None,
    ) -> TPWeightSwitchState:
        """Allocate local/full tensor aliases and reusable gather buffers."""
        if not self.supports_tp_weight_switch:
            raise RuntimeError(
                f"{type(self).__name__} does not support TP weight switching."
            )

        gather_specs = self.get_tp_weight_switch_specs(layer)
        if not gather_specs:
            raise RuntimeError(
                f"{type(self).__name__} did not declare TP weight switch specs."
            )

        state = TPWeightSwitchState()
        for spec in gather_specs:
            state.gather_parts[spec.attr_name] = self._create_tp_weight_gather_part(
                layer,
                spec,
                tp_size,
                pool=pool,
                pool_key_prefix=pool_key_prefix,
            )

        return state

    @staticmethod
    def _device_group(group: Any) -> Any:
        return getattr(group, "device_group", group)

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
        try:
            for part in state.gather_parts.values():
                if part.prepare is not None:
                    part.prepare()
                else:
                    dim = part.spec.gather_dim
                    dim = dim if dim >= 0 else part.gather_source.dim() + dim
                    if part.gather_input.data_ptr() != part.gather_source.data_ptr():
                        part.gather_input.copy_(
                            torch.movedim(part.gather_source, dim, 0)
                        )
                handle = torch.distributed.all_gather_into_tensor(
                    part.gather_output,
                    part.gather_input,
                    group=self._device_group(group),
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
                if part.finalize is not None:
                    part.finalize()
                    continue
                dim = part.spec.gather_dim
                dim = dim if dim >= 0 else part.gather_source.dim() + dim
                if dim != 0 and part.spec.make_full_contiguous:
                    part.full_tensor.copy_(torch.movedim(part.gather_output, 0, dim))
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

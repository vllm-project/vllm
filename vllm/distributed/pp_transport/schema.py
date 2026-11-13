# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-field schemas for the existing PP tensor-dict channel."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

import torch

from vllm.sequence import IntermediateTensors
from vllm.utils.math_utils import cdiv


class PPTransportDataType(StrEnum):
    """Built-in field names; downstream fields can use any nonempty string."""

    HIDDEN_STATES = "hidden_states"
    RESIDUAL = "residual"


@dataclass(frozen=True)
class PPTransportTensor:
    """Model binding; shape excludes the leading token dimension.

    Sequence-parallel fields hold ceil(global_tokens / tp_size) local rows.
    """

    shape: tuple[int, ...] = ()
    dtype: torch.dtype | None = None
    sequence_parallel: bool = False

    def num_rows(self, num_tokens: int, tp_size: int) -> int:
        """Override for device-specific token padding, shared by all buffer views."""
        return cdiv(num_tokens, tp_size) if self.sequence_parallel else num_tokens


class PPTransportSchemaInterface(Protocol):
    """Runner-facing contract, also implementable outside vLLM."""

    @property
    def has_sharded_tensors(self) -> bool: ...

    @property
    def all_gather_policy(self) -> dict[str, bool]: ...

    def make_empty_intermediate_tensors(
        self, batch_size: int, dtype: torch.dtype, device: torch.device
    ) -> IntermediateTensors: ...

    def prepare_inputs(
        self,
        buffers: IntermediateTensors,
        num_tokens: int,
        received: IntermediateTensors | None = None,
        *,
        num_received_tokens: int | None = None,
    ) -> IntermediateTensors: ...

    def make_intermediate_tensors(
        self, num_tokens: int, **tensors: torch.Tensor
    ) -> IntermediateTensors: ...


class PPTransportSchema:
    """Per-model tensor layouts, independent of the communication backend."""

    def __init__(
        self,
        tensors: Mapping[str, PPTransportTensor] | None = None,
        *,
        tp_size: int = 1,
    ):
        if tp_size < 1:
            raise ValueError("PP transport requires a positive TP size")
        self.tp_size = tp_size
        self._tensors: dict[str, PPTransportTensor] = {}
        self._sealed = False
        for name, tensor in (tensors or {}).items():
            self.register_tensor(name, tensor)

    def register_tensor(self, name: str, tensor: PPTransportTensor) -> None:
        """Register a model-local field before allocating or using its buffers."""
        if self._sealed:
            raise RuntimeError("PP fields cannot change after buffer initialization")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("PP field names must be nonempty strings")
        if name in self._tensors:
            raise ValueError(f"PP field already registered: {name}")
        self._tensors[name] = tensor

    @property
    def has_sharded_tensors(self) -> bool:
        return self.tp_size > 1 and any(
            tensor.sequence_parallel for tensor in self._tensors.values()
        )

    @property
    def all_gather_policy(self) -> dict[str, bool]:
        return {
            key: False
            for key, tensor in self._tensors.items()
            if tensor.sequence_parallel
        }

    def make_empty_intermediate_tensors(
        self, batch_size: int, dtype: torch.dtype, device: torch.device
    ) -> IntermediateTensors:
        self._sealed = True
        return _allocate_intermediate_tensors(
            self._tensors.items(), batch_size, dtype, device, self.tp_size
        )

    def prepare_inputs(
        self,
        buffers: IntermediateTensors,
        num_tokens: int,
        received: IntermediateTensors | None = None,
        *,
        num_received_tokens: int | None = None,
    ) -> IntermediateTensors:
        """Use the same per-field views during profiling, capture and execution."""
        self._sealed = True
        views = {}
        for key, buffer in buffers.items():
            binding = self._tensors.get(key)
            sharded = binding is not None and binding.sequence_parallel
            num_rows = (
                binding.num_rows(num_tokens, self.tp_size)
                if binding is not None
                else num_tokens
            )
            if num_rows > buffer.shape[0]:
                raise ValueError(f"PP input exceeds buffer capacity: {key}")
            view = buffer[:num_rows]
            if received is not None:
                source = received[key]
                copy_rows = (
                    num_rows
                    if sharded or num_received_tokens is None
                    else (
                        binding.num_rows(num_received_tokens, self.tp_size)
                        if binding is not None
                        else num_received_tokens
                    )
                )
                if sharded and source.shape[0] != num_rows:
                    raise ValueError(
                        f"SP PP stages must use matching padding for {key}"
                    )
                source = source[:copy_rows]
                if source.shape != (copy_rows, *view.shape[1:]):
                    raise ValueError(
                        f"Invalid PP input shape for {key}: {source.shape}"
                    )
                if (
                    binding is not None
                    and binding.dtype is not None
                    and source.dtype != binding.dtype
                ):
                    raise ValueError(
                        f"Invalid PP input dtype for {key}: {source.dtype}"
                    )
                # Legacy models can send FP32 residuals into model-dtype buffers.
                if (
                    torch.compiler.is_compiling()
                    or source.data_ptr() != view.data_ptr()
                ):
                    view[:copy_rows].copy_(source)
            views[key] = view
        return IntermediateTensors(views)

    def make_intermediate_tensors(
        self, num_tokens: int, **tensors: torch.Tensor
    ) -> IntermediateTensors:
        self._sealed = True
        output = {}
        for key, binding in self._tensors.items():
            num_rows = binding.num_rows(num_tokens, self.tp_size)
            tensor = tensors[key]
            if tensor.shape != (num_rows, *binding.shape):
                raise ValueError(f"Invalid PP output shape for {key}: {tensor.shape}")
            output[key] = tensor
        return IntermediateTensors(output)


def _allocate_intermediate_tensors(
    bindings: Iterable[tuple[str, PPTransportTensor]],
    num_tokens: int,
    dtype: torch.dtype,
    device: torch.device,
    tp_size: int = 1,
) -> IntermediateTensors:
    return IntermediateTensors(
        {
            key: torch.zeros(
                (
                    binding.num_rows(num_tokens, tp_size),
                    *binding.shape,
                ),
                dtype=binding.dtype or dtype,
                device=device,
            )
            for key, binding in bindings
        }
    )


def make_empty_intermediate_tensors_factory(
    keys: list[str],
    hidden_size: int,
    buffers: Mapping[str, torch.Tensor] | None = None,
):
    """Compatibility entry point for models with token-major base tensors."""
    bindings = {key: PPTransportTensor((hidden_size,)) for key in keys}
    bindings.update(
        {
            key: PPTransportTensor(tuple(buffer.shape[1:]), dtype=buffer.dtype)
            for key, buffer in (buffers or {}).items()
        }
    )

    def make_empty_intermediate_tensors(batch_size, dtype, device):
        return _allocate_intermediate_tensors(
            bindings.items(), batch_size, dtype, device
        )

    return make_empty_intermediate_tensors


def copy_intermediate_tensors(
    buffers: IntermediateTensors, tensors: IntermediateTensors
) -> IntermediateTensors:
    """Copy capture outputs and retain their individual shapes for graph replay."""
    views = {}
    for key, tensor in tensors.items():
        buffer = buffers[key]
        if (
            tensor.ndim == 0
            or tensor.ndim != buffer.ndim
            or tensor.shape[1:] != buffer.shape[1:]
            or tensor.shape[0] > buffer.shape[0]
            or tensor.dtype != buffer.dtype
        ):
            raise ValueError(
                f"Invalid {key}: {tensor.shape}/{tensor.dtype} "
                f"for buffer {buffer.shape}/{buffer.dtype}"
            )
        buffer[: tensor.shape[0]].copy_(tensor)
        views[key] = buffer[: tensor.shape[0]]
    return IntermediateTensors(views)

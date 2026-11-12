# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-field schemas for the existing PP tensor-dict channel."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum

import torch

from vllm.sequence import IntermediateTensors
from vllm.utils.math_utils import cdiv


class PPTransportDataType(StrEnum):
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


class PPTransportSchema:
    """Per-model tensor layouts, independent of the communication backend."""

    def __init__(
        self,
        tensors: Mapping[PPTransportDataType, PPTransportTensor] | None = None,
        *,
        tp_size: int = 1,
    ):
        if tp_size < 1:
            raise ValueError("PP transport requires a positive TP size")
        self.tp_size = tp_size
        self._tensors = tuple(
            (PPTransportDataType(data_type).value, tensor)
            for data_type, tensor in (tensors or {}).items()
        )
        self._input_partitions = {
            key: tp_size for key, tensor in self._tensors if tensor.sequence_parallel
        }
        self._input_dtypes = {
            key: tensor.dtype
            for key, tensor in self._tensors
            if tensor.dtype is not None
        }
        self.has_sharded_tensors = tp_size > 1 and any(
            tensor.sequence_parallel for _, tensor in self._tensors
        )
        self.all_gather_policy = {
            key: False for key, tensor in self._tensors if tensor.sequence_parallel
        }

    def make_empty_intermediate_tensors(
        self, batch_size: int, dtype: torch.dtype, device: torch.device
    ) -> IntermediateTensors:
        return _allocate_intermediate_tensors(
            self._tensors, batch_size, dtype, device, self.tp_size
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
        views = {}
        for key, buffer in buffers.items():
            partition = self._input_partitions.get(key, 1)
            num_rows = cdiv(num_tokens, partition)
            if num_rows > buffer.shape[0]:
                raise ValueError(f"PP input exceeds buffer capacity: {key}")
            view = buffer[:num_rows]
            if received is not None:
                source = received[key]
                copy_rows = (
                    num_rows
                    if partition > 1 or num_received_tokens is None
                    else num_received_tokens
                )
                if partition > 1 and source.shape[0] != num_rows:
                    raise ValueError(
                        f"SP PP stages must use matching padding for {key}"
                    )
                source = source[:copy_rows]
                if source.shape != (copy_rows, *view.shape[1:]):
                    raise ValueError(
                        f"Invalid PP input shape for {key}: {source.shape}"
                    )
                if (
                    key in self._input_dtypes
                    and source.dtype != self._input_dtypes[key]
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
        output = {}
        for key, binding in self._tensors:
            partition = self.tp_size if binding.sequence_parallel else 1
            num_rows = cdiv(num_tokens, partition)
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
                    cdiv(num_tokens, tp_size)
                    if binding.sequence_parallel
                    else num_tokens,
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

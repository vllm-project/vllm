# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-field schemas for the existing PP tensor-dict channel."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import torch

from vllm.distributed.pp_transport.fields import PPTransportField
from vllm.sequence import IntermediateTensors
from vllm.utils.math_utils import cdiv


@dataclass(frozen=True)
class PPTransportTensorSpec:
    """Tensor layout; shape excludes the leading token dimension.

    Sequence-parallel fields hold ceil(global_tokens / tp_size) local rows.
    """

    shape: tuple[int, ...] = ()
    dtype: torch.dtype | None = None
    sequence_parallel: bool = False

    def num_rows(self, num_tokens: int, tp_size: int) -> int:
        """Override for device-specific token padding, shared by all buffer views."""
        return cdiv(num_tokens, tp_size) if self.sequence_parallel else num_tokens

    def allocate(
        self, num_tokens: int, dtype: torch.dtype, device: torch.device, tp_size: int
    ) -> torch.Tensor:
        """Override to bind a model-owned persistent receive buffer."""
        return torch.zeros(
            (self.num_rows(num_tokens, tp_size), *self.shape),
            dtype=self.dtype or dtype,
            device=device,
        )


class PPTransportSchema:
    """Per-model tensor layouts, independent of the communication backend."""

    def __init__(
        self,
        specs: Mapping[PPTransportField | str, PPTransportTensorSpec] | None = None,
        *,
        tp_size: int = 1,
    ):
        if tp_size < 1:
            raise ValueError("PP transport requires a positive TP size")
        self.tp_size = tp_size
        self._recv_specs: dict[str, PPTransportTensorSpec] = {}
        self._send_specs: dict[str, PPTransportTensorSpec] = {}
        for name, spec in (specs or {}).items():
            self.register_tensor(name, spec)

    def register_tensor(
        self,
        name: PPTransportField | str,
        spec: PPTransportTensorSpec,
        *,
        receive: bool = True,
        send: bool = True,
    ) -> None:
        """Register a model-local field before allocating or using its buffers."""
        if not isinstance(name, str) or not name.strip():
            raise ValueError("PP field names must be nonempty strings")
        name = str(name)
        if not receive and not send:
            raise ValueError("PP field must be received or sent")
        if (receive and name in self._recv_specs) or (
            send and name in self._send_specs
        ):
            raise ValueError(f"PP field already registered: {name}")
        if receive:
            self._recv_specs[name] = spec
        if send:
            self._send_specs[name] = spec

    @property
    def recv_specs(self) -> Mapping[str, PPTransportTensorSpec]:
        """Read-only live view of explicitly registered receive fields."""
        return MappingProxyType(self._recv_specs)

    @property
    def send_specs(self) -> Mapping[str, PPTransportTensorSpec]:
        """Read-only live view of explicitly registered send fields."""
        return MappingProxyType(self._send_specs)

    @property
    def has_sequence_parallel_tensors(self) -> bool:
        return self.tp_size > 1 and any(
            spec.sequence_parallel
            for spec in (*self._recv_specs.values(), *self._send_specs.values())
        )

    @property
    def recv_all_gather_tensors(self) -> dict[str, bool]:
        return {
            key: False
            for key, spec in self._recv_specs.items()
            if spec.sequence_parallel
        }

    @property
    def send_all_gather_tensors(self) -> dict[str, bool]:
        return {
            key: False
            for key, spec in self._send_specs.items()
            if spec.sequence_parallel
        }

    def make_empty_intermediate_tensors(
        self, batch_size: int, dtype: torch.dtype, device: torch.device
    ) -> IntermediateTensors:
        return IntermediateTensors(
            {
                key: spec.allocate(batch_size, dtype, device, self.tp_size)
                for key, spec in self._recv_specs.items()
            }
        )

    def prepare_outputs(
        self, tensors: IntermediateTensors, num_tokens: int
    ) -> IntermediateTensors:
        """Trim full fields while retaining the padded rows of SP fields."""
        views = {}
        for key, tensor in tensors.items():
            spec = self._send_specs.get(key, PPTransportTensorSpec())
            if spec.sequence_parallel and self.tp_size > 1:
                views[key] = tensor
            else:
                views[key] = tensor[: spec.num_rows(num_tokens, self.tp_size)]
        return IntermediateTensors(views)

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
            spec = self._recv_specs.get(key, PPTransportTensorSpec())
            num_rows = spec.num_rows(num_tokens, self.tp_size)
            if num_rows > buffer.shape[0]:
                raise ValueError(f"PP input exceeds buffer capacity: {key}")
            view = views[key] = buffer[:num_rows]
            if received is None:
                continue

            source = received[key]
            sharded = spec.sequence_parallel and self.tp_size > 1
            if sharded and source.shape[0] != num_rows:
                raise ValueError(f"SP PP stages must use matching padding for {key}")
            copy_tokens = num_tokens
            if not sharded and num_received_tokens is not None:
                copy_tokens = num_received_tokens
            copy_rows = spec.num_rows(copy_tokens, self.tp_size)
            source = source[:copy_rows]
            if source.shape != (copy_rows, *view.shape[1:]):
                raise ValueError(f"Invalid PP input shape for {key}: {source.shape}")
            if spec.dtype is not None and source.dtype != spec.dtype:
                raise ValueError(f"Invalid PP input dtype for {key}: {source.dtype}")
            # Legacy models can send FP32 residuals into model-dtype buffers.
            if torch.compiler.is_compiling() or source.data_ptr() != view.data_ptr():
                view[:copy_rows].copy_(source)
        return IntermediateTensors(views)

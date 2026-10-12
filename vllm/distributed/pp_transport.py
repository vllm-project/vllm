# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Typed side inputs carried by the existing PP tensor-dict transport.

Multimodal embedding entries contain encoder outputs, not weights. Drafters
retain their local text embedding lookup.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import TypeVar, overload

import torch

from vllm.sequence import IntermediateTensors

_T = TypeVar("_T")


class PPTransportDataType(str, Enum):
    HIDDEN_STATES = "hidden_states"
    RESIDUAL = "residual"
    TOPK_INDICES = "pp_transport.topk_indices"
    MULTIMODAL_EMBEDDINGS = "pp_transport.multimodal_embeddings"
    MULTIMODAL_MASK = "pp_transport.multimodal_mask"


@dataclass(frozen=True)
class PPTransportDataSpec:
    """Wire policy, shared by senders, receivers and model initialization."""

    model_input: bool = False
    all_gather: bool | None = False
    indexed: bool = False
    companions: tuple[PPTransportDataType, ...] = ()


PP_TRANSPORT_REGISTRY: Mapping[PPTransportDataType, PPTransportDataSpec] = (
    MappingProxyType(
        {
            PPTransportDataType.HIDDEN_STATES: PPTransportDataSpec(
                model_input=True, all_gather=None
            ),
            PPTransportDataType.RESIDUAL: PPTransportDataSpec(
                model_input=True, all_gather=None
            ),
            PPTransportDataType.TOPK_INDICES: PPTransportDataSpec(model_input=True),
            PPTransportDataType.MULTIMODAL_EMBEDDINGS: PPTransportDataSpec(
                indexed=True, companions=(PPTransportDataType.MULTIMODAL_MASK,)
            ),
            PPTransportDataType.MULTIMODAL_MASK: PPTransportDataSpec(),
        }
    )
)


def _get_data_type(key: str) -> PPTransportDataType | None:
    for data_type, spec in PP_TRANSPORT_REGISTRY.items():
        if key == data_type.value or (
            spec.indexed and key.startswith(data_type.value + ".")
        ):
            return data_type
    if key.startswith("pp_transport."):
        raise ValueError(f"Unregistered PP transport field: {key}")
    return None


@dataclass(frozen=True)
class PPTransportTensor:
    """Model binding; shape excludes the leading token dimension.

    A buffer supplies both shape and dtype. Received values are copied into it
    without replacing its address, and outgoing values use its active rows.
    """

    shape: tuple[int, ...] = ()
    dtype: torch.dtype | None = None
    buffer: torch.Tensor | None = None
    receive: bool = True
    send: bool = True

    def __post_init__(self):
        if self.buffer is not None:
            if self.buffer.ndim == 0:
                raise ValueError("PP buffers must have a token dimension")
            if self.shape and self.shape != self.buffer.shape[1:]:
                raise ValueError("PP tensor shape does not match its buffer")
            if self.dtype is not None and self.dtype != self.buffer.dtype:
                raise ValueError("PP tensor dtype does not match its buffer")
            object.__setattr__(self, "shape", tuple(self.buffer.shape[1:]))
            object.__setattr__(self, "dtype", self.buffer.dtype)


class PPTransport:
    """Per-model selection of registered model inputs and graph-external payloads."""

    def __init__(
        self,
        tensors: Mapping[PPTransportDataType, PPTransportTensor] | None = None,
        *,
        payload_types: Sequence[PPTransportDataType] = (),
    ):
        receive = []
        send = []
        for data_type, tensor in (tensors or {}).items():
            if not PP_TRANSPORT_REGISTRY[data_type].model_input:
                raise ValueError(f"{data_type.value} belongs in a PP payload")
            if tensor.receive:
                receive.append((data_type.value, tensor))
            if tensor.send:
                send.append((data_type.value, tensor))
        self._receive = tuple(receive)
        self._send = tuple(send)
        selected = set(payload_types)
        for data_type in payload_types:
            selected.update(PP_TRANSPORT_REGISTRY[data_type].companions)
        for data_type in selected:
            if PP_TRANSPORT_REGISTRY[data_type].model_input:
                raise ValueError(f"{data_type.value} belongs in the model input schema")
        self._payload_types = frozenset(selected)

    def make_empty_intermediate_tensors(
        self, batch_size: int, dtype: torch.dtype, device: torch.device
    ) -> IntermediateTensors:
        return IntermediateTensors(
            {
                key: torch.zeros(
                    (batch_size, *tensor.shape),
                    dtype=tensor.dtype or dtype,
                    device=device,
                )
                for key, tensor in self._receive
            }
        )

    def restore_buffers(self, tensors: IntermediateTensors) -> None:
        for key, binding in self._receive:
            if binding.buffer is not None:
                _copy_to_buffer(key, tensors[key], binding.buffer)

    def make_intermediate_tensors(
        self, num_tokens: int, **tensors: torch.Tensor
    ) -> IntermediateTensors:
        output = {}
        for key, binding in self._send:
            if binding.buffer is None:
                output[key] = tensors[key]
            else:
                if num_tokens > binding.buffer.shape[0]:
                    raise ValueError(f"PP send exceeds buffer capacity: {key}")
                output[key] = binding.buffer[:num_tokens]
        return IntermediateTensors(output)

    def new_payload(
        self, tensors: Mapping[str, torch.Tensor] | None = None
    ) -> "PPTransportPayload":
        return PPTransportPayload(self._payload_types, tensors)


def make_empty_intermediate_tensors_factory(keys: list[str], hidden_size: int):
    """Compatibility entry point for models with token-major base tensors."""
    return PPTransport(
        {PPTransportDataType(key): PPTransportTensor((hidden_size,)) for key in keys}
    ).make_empty_intermediate_tensors


class PPTransportAllGatherPolicy(dict[str, bool]):
    """Use registered TP policies, retaining worker defaults for base tensors."""

    @overload
    def get(self, key: str, default: None = None) -> bool | None: ...

    @overload
    def get(self, key: str, default: bool) -> bool: ...

    @overload
    def get(self, key: str, default: _T) -> bool | _T: ...

    def get(self, key: str, default: _T | None = None) -> bool | _T | None:
        data_type = _get_data_type(key)
        if data_type is not None:
            policy = PP_TRANSPORT_REGISTRY[data_type].all_gather
            if policy is not None:
                return policy
        return super().get(key, default)


def _copy_to_buffer(key: str, tensor: torch.Tensor, buffer: torch.Tensor) -> None:
    """Restore a received token-major tensor into a persistent model buffer."""
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


class PPTransportPayload:
    """Per-forward side inputs, kept outside the target CUDA graph's inputs.

    Top-k buffers consumed inside the model stay in its IntermediateTensors
    schema. Embeddings for a later consumer can pass through intermediate
    stages without changing those stages' compiled forward signatures.
    """

    def __init__(
        self,
        data_types: frozenset[PPTransportDataType],
        tensors: Mapping[str, torch.Tensor] | None = None,
    ):
        self._data_types = data_types
        self.tensors = {
            key: tensor
            for key, tensor in (tensors or {}).items()
            if _get_data_type(key) in data_types
        }

    def set_multimodal_embeddings(
        self, embeddings: list[torch.Tensor], is_multimodal: torch.Tensor
    ) -> None:
        if PPTransportDataType.MULTIMODAL_EMBEDDINGS not in self._data_types:
            raise ValueError("Multimodal embeddings were not selected for PP transport")
        # Keep the encoder's CPU mask on CPU: draft merging uses it without D2H.
        if (
            is_multimodal.device.type != "cpu"
            or is_multimodal.dtype != torch.bool
            or is_multimodal.ndim != 1
        ):
            raise ValueError(
                "The multimodal mask must be a one-dimensional CPU bool tensor"
            )
        prefix = PPTransportDataType.MULTIMODAL_EMBEDDINGS.value + "."
        self.tensors = {
            key: tensor
            for key, tensor in self.tensors.items()
            if not key.startswith(prefix)
        }
        self.tensors[PPTransportDataType.MULTIMODAL_MASK.value] = (
            is_multimodal.contiguous()
        )
        for index, embedding in enumerate(embeddings):
            # A tensor's Python attributes are not sent by tensor-dict transport.
            modality = getattr(embedding, "modality", "")
            # MM pruning may return a noncontiguous view after stripping channels.
            self.tensors[f"{prefix}{index}.{modality}"] = embedding.contiguous()

    def get_multimodal_embeddings(
        self,
    ) -> tuple[list[torch.Tensor], torch.Tensor] | None:
        mask = self.tensors.get(PPTransportDataType.MULTIMODAL_MASK.value)
        if mask is None:
            return None
        prefix = PPTransportDataType.MULTIMODAL_EMBEDDINGS.value + "."
        entries = []
        for key, embedding in self.tensors.items():
            if key.startswith(prefix):
                encoded_index, modality = key.removeprefix(prefix).split(".", 1)
                entries.append((int(encoded_index), modality, embedding))
        entries.sort(key=lambda entry: entry[0])
        embeddings = []
        for expected, (index, modality, embedding) in enumerate(entries):
            if index != expected:
                raise ValueError("Missing or duplicate PP multimodal embedding")
            if modality:
                embedding.modality = modality
            embeddings.append(embedding)
        return embeddings, mask

    def relay(self, output: IntermediateTensors) -> IntermediateTensors:
        """Carry side inputs without replacing tensors produced by this stage."""
        if not self.tensors:
            return output
        return IntermediateTensors(self.tensors | output.tensors)

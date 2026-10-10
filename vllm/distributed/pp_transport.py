# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Typed side inputs carried by the existing PP tensor-dict transport.

Embedding entries contain activations, not weights. Drafters retain their local
embedding lookup for newly sampled tokens.
"""

from collections.abc import Mapping
from enum import Enum
from typing import TypeVar, overload

import torch

from vllm.sequence import IntermediateTensors

_T = TypeVar("_T")


class PPTransportDataType(str, Enum):
    TOPK_INDICES = "pp_transport.topk_indices"
    TARGET_EMBEDDINGS = "pp_transport.target_embeddings"
    MULTIMODAL_EMBEDDINGS = "pp_transport.multimodal_embeddings"
    MULTIMODAL_MASK = "pp_transport.multimodal_mask"


class PPTransportAllGatherPolicy(dict[str, bool]):
    """Side inputs may be TP-local; never reconstruct them from other TP lanes."""

    @overload
    def get(self, key: str, default: None = None) -> bool | None: ...

    @overload
    def get(self, key: str, default: bool) -> bool: ...

    @overload
    def get(self, key: str, default: _T) -> bool | _T: ...

    def get(self, key: str, default: _T | None = None) -> bool | _T | None:
        if key.startswith("pp_transport."):
            return False
        return super().get(key, default)


def add_pp_transport_tensor(
    tensors: IntermediateTensors,
    data_type: PPTransportDataType,
    tensor: torch.Tensor,
) -> IntermediateTensors:
    """Attach a model-owned tensor. The sender retains it until send completion."""
    tensors[data_type.value] = tensor
    return tensors


def copy_pp_transport_tensor(
    tensors: IntermediateTensors,
    data_type: PPTransportDataType,
    buffer: torch.Tensor,
) -> None:
    """Restore a received token-major tensor into a persistent model buffer."""
    tensor = tensors[data_type.value]
    if (
        tensor.ndim == 0
        or tensor.ndim != buffer.ndim
        or tensor.shape[1:] != buffer.shape[1:]
        or tensor.shape[0] > buffer.shape[0]
        or tensor.dtype != buffer.dtype
    ):
        raise ValueError(
            f"Invalid {data_type.value}: {tensor.shape}/{tensor.dtype} "
            f"for buffer {buffer.shape}/{buffer.dtype}"
        )
    buffer[: tensor.shape[0]].copy_(tensor)


class PPTransportPayload:
    """Per-forward side inputs, kept outside the target CUDA graph's inputs.

    Top-k buffers consumed inside the model stay in its IntermediateTensors
    schema. Embeddings for a later consumer can pass through intermediate
    stages without changing those stages' compiled forward signatures.
    """

    def __init__(self, tensors: Mapping[str, torch.Tensor] | None = None):
        self.tensors = {
            key: tensor
            for key, tensor in (tensors or {}).items()
            if key.startswith("pp_transport.")
            and key != PPTransportDataType.TOPK_INDICES.value
        }

    def set_target_embeddings(self, embeddings: torch.Tensor) -> None:
        """Carry activations, not the embedding weight matrix."""
        self.tensors[PPTransportDataType.TARGET_EMBEDDINGS.value] = (
            embeddings.contiguous()
        )

    def get_target_embeddings(self) -> torch.Tensor | None:
        return self.tensors.get(PPTransportDataType.TARGET_EMBEDDINGS.value)

    def set_multimodal_embeddings(
        self, embeddings: list[torch.Tensor], is_multimodal: torch.Tensor
    ) -> None:
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

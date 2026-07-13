# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Data classes for the hidden-state Mooncake Store EC connector."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorMetadata
from vllm.distributed.ec_transfer.ec_connector.mooncake_store_hidden.keys import (
    escape_key_part,
    make_hidden_data_key,
)

HIDDEN_OBJECT_KIND = "encoder_output"
HIDDEN_STORAGE_LAYOUT = "replicated_object"
HIDDEN_TENSOR_LAYOUT = "tensor"
HIDDEN_PROTOCOL_VERSION = "v1"
MOONCAKE_TENSOR_METADATA_NBYTES = 304


@dataclass(frozen=True)
class HiddenKeyMetadata:
    """Metadata that defines the semantic namespace for hidden reuse."""

    cache_prefix: str
    kind: str
    model_name: str
    encoder: str
    storage: str
    parallel: str
    tensor_layout: str


@dataclass(frozen=True, order=True)
class HiddenPoolKey:
    """Key for addressing one hidden tensor in the distributed store."""

    key_metadata: HiddenKeyMetadata
    identifier: str

    def to_string(self) -> str:
        meta = self.key_metadata
        prefix = (
            f"{escape_key_part(meta.cache_prefix)}@" if meta.cache_prefix else ""
        )
        return (
            f"{prefix}hidden"
            f"@kind:{escape_key_part(meta.kind)}"
            f"@model:{escape_key_part(meta.model_name)}"
            f"@encoder:{escape_key_part(meta.encoder)}"
            f"@storage:{escape_key_part(meta.storage)}"
            f"@parallel:{escape_key_part(meta.parallel)}"
            f"@tensor_layout:{escape_key_part(meta.tensor_layout)}"
            f"@id:{escape_key_part(self.identifier)}"
        )


@dataclass
class MMMeta:
    """Per hidden object metadata passed from scheduler to worker."""

    identifier: str
    modality: str | None = None
    can_save: bool = False
    load_spec: LoadSpec | None = None


@dataclass(frozen=True)
class TensorMeta:
    """Canonical contiguous tensor descriptor for one hidden store object."""

    pool_key: HiddenPoolKey
    protocol_version: str
    layout: str
    shape: tuple[int, ...]
    dtype: str
    nbytes: int
    device_type: str
    data_offset: int = MOONCAKE_TENSOR_METADATA_NBYTES
    producer_stage: str = "encoder"


@dataclass
class LoadSpec:
    """Specification for loading a hidden tensor from external store."""

    can_load: bool = False


@dataclass
class HiddenSaveRequest:
    """Specification for asynchronously storing one hidden tensor."""

    pool_key: HiddenPoolKey
    tensor: torch.Tensor
    with_soft_pin: bool = False

    @property
    def identifier(self) -> str:
        return self.pool_key.identifier


@dataclass
class MooncakeStoreConnectorMetadata(ECConnectorMetadata):
    """Metadata passed from scheduler to worker for hidden store operations."""

    items: list[MMMeta] = field(default_factory=list)

    def add_item(self, item: MMMeta) -> None:
        self.items.append(item)


@dataclass
class HiddenStoreOperationStats:
    """Minimal per-operation telemetry aligned with Mooncake KV store stats."""

    data: dict[str, list[dict[str, int | float | str]]] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not self.data

    def record_operation(
        self,
        operation: str,
        duration_seconds: float,
        num_keys: int,
        *,
        num_bytes: int = 0,
        status: str = "ok",
        num_failed_keys: int = 0,
    ) -> None:
        self.data.setdefault(operation, []).append(
            {
                "duration_seconds": duration_seconds,
                "num_keys": num_keys,
                "num_bytes": num_bytes,
                "status": status,
                "num_failed_keys": num_failed_keys,
            }
        )


class HiddenTensorDatabase:
    """Maps hidden tensors to store keys and GPU memory descriptors."""

    def prepare_value(
        self,
        pool_key: HiddenPoolKey,
        tensor: torch.Tensor,
    ) -> tuple[str, list[int], list[int]]:
        return (
            make_hidden_data_key(pool_key),
            [tensor.data_ptr()],
            [tensor.numel() * tensor.element_size()],
        )


def build_tensor_meta(
    pool_key: HiddenPoolKey,
    tensor: torch.Tensor,
) -> TensorMeta:
    """Build metadata for the canonical stored hidden tensor layout."""
    if not tensor.is_contiguous():
        raise ValueError("Hidden tensor descriptor requires a contiguous tensor")

    return TensorMeta(
        pool_key=pool_key,
        protocol_version=HIDDEN_PROTOCOL_VERSION,
        layout=HIDDEN_TENSOR_LAYOUT,
        shape=tuple(tensor.shape),
        dtype=str(tensor.dtype),
        nbytes=tensor.numel() * tensor.element_size(),
        device_type=tensor.device.type,
        data_offset=MOONCAKE_TENSOR_METADATA_NBYTES,
    )


def validate_loaded_tensor(tensor: torch.Tensor, meta: TensorMeta) -> None:
    if tuple(tensor.shape) != tuple(meta.shape):
        raise ValueError(
            "Hidden tensor shape mismatch: "
            f"actual={tuple(tensor.shape)} expected={meta.shape}"
        )

    if str(tensor.dtype) != meta.dtype:
        raise ValueError(
            "Hidden tensor dtype mismatch: "
            f"actual={tensor.dtype} expected={meta.dtype}"
        )

    actual_nbytes = tensor.numel() * tensor.element_size()
    if actual_nbytes != meta.nbytes:
        raise ValueError(
            "Hidden tensor nbytes mismatch: "
            f"actual={actual_nbytes} expected={meta.nbytes}"
        )

    if meta.layout != HIDDEN_TENSOR_LAYOUT:
        raise ValueError(f"Unsupported hidden tensor layout: {meta.layout}")

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Routed-experts artifact encoding and block publication."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

import numpy as np

from vllm.distributed.artifact_connector.store import (
    ArtifactCorruptionError,
    ArtifactObject,
    ArtifactReader,
    ArtifactStore,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.artifact_connector.buffer import (
        RoutedExpertsArtifactBuffer,
    )


def get_routing_shape_and_dtype(
    vllm_config: VllmConfig,
) -> tuple[tuple[int, int], str]:
    hf_config = vllm_config.model_config.hf_text_config
    num_experts = vllm_config.model_config.get_num_experts()
    return (
        (
            hf_config.num_hidden_layers,
            vllm_config.model_config.get_num_experts_per_token(),
        ),
        "uint8" if num_experts <= 256 else "uint16",
    )


def routed_experts_key(block_hash: bytes, artifact_namespace: str) -> str:
    digest = hashlib.sha256()
    digest.update(artifact_namespace.encode())
    digest.update(b"\0")
    digest.update(block_hash)
    return f"vllm-artifact/{digest.hexdigest()}"


def encode_routed_experts_array(array: np.ndarray) -> bytes:
    return np.ascontiguousarray(array).tobytes(order="C")


def decode_routed_experts_array(
    payload: bytes,
    *,
    shape_per_token: tuple[int, ...],
    dtype: np.dtype[Any],
) -> np.ndarray:
    row_nbytes = int(np.prod(shape_per_token)) * dtype.itemsize
    if not payload or len(payload) % row_nbytes:
        raise ArtifactCorruptionError("invalid routed-experts artifact size")
    try:
        return np.frombuffer(payload, dtype=dtype).reshape((-1, *shape_per_token))
    except ValueError as error:
        raise ArtifactCorruptionError("invalid routed-experts artifact") from error


def materialize_routed_experts(
    store: ArtifactReader,
    artifact_keys: list[str],
    *,
    shape_per_token: tuple[int, ...],
    dtype: np.dtype[Any],
    rows_per_object: int,
) -> np.ndarray:
    if not artifact_keys:
        raise ValueError("routed-experts artifact key list must not be empty")
    payloads = store.get(artifact_keys)
    if len(payloads) != len(artifact_keys):
        raise ArtifactCorruptionError(
            "artifact backend returned the wrong object count"
        )
    arrays = [
        decode_routed_experts_array(
            payload,
            shape_per_token=shape_per_token,
            dtype=dtype,
        )
        for payload in payloads
    ]
    if any(len(array) != rows_per_object for array in arrays):
        raise ArtifactCorruptionError("artifact object has an invalid row count")
    return arrays[0] if len(arrays) == 1 else np.concatenate(arrays)


class RoutedExpertsRequestCore:
    """Publish immutable full R3 blocks and release captured rows."""

    def __init__(
        self,
        store: ArtifactStore,
        source: RoutedExpertsArtifactBuffer,
    ) -> None:
        self._store = store
        self._source = source

    def commit(
        self,
        *,
        request_id: str,
        artifact_namespace: str,
        block_hashes: list[bytes],
        block_start: int,
        block_size: int,
    ) -> None:
        if block_start < 0 or block_start % block_size or not block_hashes:
            raise ValueError("invalid artifact full-block commit")
        block_end = block_start + len(block_hashes) * block_size
        array = self._source.read(request_id, block_start, block_end)
        objects = []
        for index, block_hash in enumerate(block_hashes):
            local_start = index * block_size
            objects.append(
                ArtifactObject(
                    key=routed_experts_key(block_hash, artifact_namespace),
                    payload=encode_routed_experts_array(
                        array[local_start : local_start + block_size]
                    ),
                )
            )
        self._store.put(objects)
        self._source.release_through(request_id, block_end)

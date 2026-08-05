# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Routed-experts artifact keys, publication, and materialization."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from vllm.distributed.artifact_connector.store import (
    ArtifactObject,
    ArtifactReader,
    ArtifactStore,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig


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
    object_size = rows_per_object * math.prod(shape_per_token) * dtype.itemsize
    payload = store.get_concatenated(
        artifact_keys,
        object_size=object_size,
    )
    return np.frombuffer(payload, dtype=dtype).reshape((-1, *shape_per_token))


def publish_routed_experts(
    store: ArtifactStore,
    *,
    batches: list[tuple[Sequence[str], list[tuple[int, np.ndarray]]]],
    block_size: int,
) -> None:
    """Publish immutable full R3 blocks."""
    objects = []
    for artifact_keys, blocks in batches:
        for block_start, array in blocks:
            if block_start < 0 or block_start % block_size:
                raise ValueError("artifact block start is not hash-block aligned")
            block_index = block_start // block_size
            if block_index >= len(artifact_keys):
                raise ValueError(
                    "artifact block has no corresponding key: "
                    f"start={block_start}, index={block_index}, "
                    f"keys={len(artifact_keys)}, block_size={block_size}"
                )
            if len(array) != block_size:
                raise ValueError("artifact block length does not match hash block size")
            objects.append(
                ArtifactObject(
                    key=artifact_keys[block_index],
                    payload=array.tobytes(order="C"),
                )
            )
    store.put(objects)

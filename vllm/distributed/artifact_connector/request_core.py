# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backend-independent request lifecycle and object format."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Sequence
from dataclasses import dataclass
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

_SCHEMA_VERSION = 3
_FIELD_NAME = "routed_experts"
_HEADER_LENGTH = struct.Struct("<I")
_MAX_HEADER_BYTES = 4096


@dataclass(frozen=True)
class ArtifactCommit:
    request_id: str
    artifact_namespace: str
    block_hashes: Sequence[bytes]
    block_start: int
    hash_block_size: int


@dataclass(frozen=True)
class ArtifactFinalize:
    request_id: str
    artifact_namespace: str
    block_hashes: Sequence[bytes]
    tail_block_hash: bytes | None
    token_end: int
    hash_block_size: int


def _canonical_json(value: dict[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _digest(*parts: bytes) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(len(part).to_bytes(8, "big"))
        digest.update(part)
    return digest.hexdigest()


def get_routing_shape_and_dtype(
    vllm_config: VllmConfig,
) -> tuple[tuple[int, int], str]:
    """Return the logical per-token routing shape and dtype."""
    hf_config = vllm_config.model_config.hf_text_config
    num_layers = hf_config.num_hidden_layers
    moe_top_k = vllm_config.model_config.get_num_experts_per_token()
    num_experts = vllm_config.model_config.get_num_experts()
    dtype = "uint8" if num_experts <= 256 else "uint16"
    return (num_layers, moe_top_k), dtype


def encode_routed_experts_array(
    *,
    key: str,
    kind: str,
    array: np.ndarray,
    source_token_start: int,
) -> bytes:
    """Encode one self-describing immutable array object."""
    contiguous = np.ascontiguousarray(array)
    raw = contiguous.tobytes(order="C")
    header = {
        "schema_version": _SCHEMA_VERSION,
        "key": key,
        "kind": kind,
        "field": _FIELD_NAME,
        "dtype": contiguous.dtype.str,
        "shape": list(contiguous.shape),
        "source_token_start": source_token_start,
        "payload_sha256": hashlib.sha256(raw).hexdigest(),
    }
    encoded_header = _canonical_json(header)
    if len(encoded_header) > _MAX_HEADER_BYTES:
        raise ValueError("artifact object header is too large")
    return _HEADER_LENGTH.pack(len(encoded_header)) + encoded_header + raw


def decode_routed_experts_array(
    payload: bytes,
    *,
    expected_key: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Decode and validate one self-describing immutable array object."""
    if len(payload) < _HEADER_LENGTH.size:
        raise ArtifactCorruptionError("artifact object is truncated")
    (header_length,) = _HEADER_LENGTH.unpack(payload[: _HEADER_LENGTH.size])
    if header_length <= 0 or header_length > _MAX_HEADER_BYTES:
        raise ArtifactCorruptionError("invalid artifact header length")
    header_end = _HEADER_LENGTH.size + header_length
    if header_end > len(payload):
        raise ArtifactCorruptionError("artifact header is truncated")
    try:
        header = json.loads(payload[_HEADER_LENGTH.size : header_end])
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ArtifactCorruptionError("invalid artifact object header") from error
    if not isinstance(header, dict):
        raise ArtifactCorruptionError("artifact object header must be an object")
    if (
        header.get("schema_version") != _SCHEMA_VERSION
        or header.get("key") != expected_key
        or header.get("kind") not in ("block", "tail")
        or header.get("field") != _FIELD_NAME
    ):
        raise ArtifactCorruptionError("artifact object identity mismatch")
    raw = payload[header_end:]
    if header.get("payload_sha256") != hashlib.sha256(raw).hexdigest():
        raise ArtifactCorruptionError("artifact payload checksum mismatch")
    try:
        dtype = np.dtype(header["dtype"])
        shape = tuple(int(value) for value in header["shape"])
        source_token_start = int(header["source_token_start"])
    except (KeyError, TypeError, ValueError) as error:
        raise ArtifactCorruptionError("invalid artifact array metadata") from error
    if (
        not shape
        or any(dimension < 0 for dimension in shape)
        or dtype.kind not in "biuf"
        or shape[0] <= 0
        or source_token_start < 0
        or math.prod(shape) * dtype.itemsize != len(raw)
    ):
        raise ArtifactCorruptionError("invalid artifact array shape")
    try:
        array = np.frombuffer(raw, dtype=dtype).reshape(shape)
    except (TypeError, ValueError) as error:
        raise ArtifactCorruptionError("invalid artifact array payload") from error
    return array, header


def routed_experts_key(block_hash: bytes, artifact_namespace: str) -> str:
    """Derive an R3 key from a cache generation and KV-compatible hash."""
    return f"vllm-artifact/{_digest(artifact_namespace.encode(), block_hash)}"


def materialize_routed_experts(
    store: ArtifactReader,
    artifact_keys: list[str],
    *,
    expected_shape_per_token: tuple[int, ...] | None = None,
    expected_dtype: np.dtype[Any] | None = None,
    expected_token_start: int = 0,
    expected_token_end: int | None = None,
    hash_block_size: int | None = None,
) -> np.ndarray:
    """Read ordered keys and materialize complete routed-experts rows."""
    if not artifact_keys:
        raise ValueError("routed-experts artifact key list must not be empty")
    payloads = store.get(artifact_keys)
    if len(payloads) != len(artifact_keys):
        raise ArtifactCorruptionError(
            "artifact backend returned the wrong object count"
        )
    arrays: list[np.ndarray] = []
    next_start = expected_token_start
    shape_per_token: tuple[int, ...] | None = None
    dtype: np.dtype[Any] | None = None
    for key, payload in zip(artifact_keys, payloads, strict=True):
        array, header = decode_routed_experts_array(
            payload,
            expected_key=key,
        )
        source_start = int(header["source_token_start"])
        if source_start != next_start:
            raise ArtifactCorruptionError(
                "artifact keys do not cover one contiguous logical range"
            )
        if expected_token_end is not None:
            if hash_block_size is None or hash_block_size <= 0:
                raise ValueError("hash_block_size must be positive")
            expected_rows = min(hash_block_size, expected_token_end - next_start)
            expected_kind = "block" if expected_rows == hash_block_size else "tail"
            if expected_rows <= 0 or len(array) != expected_rows:
                raise ArtifactCorruptionError(
                    "artifact object does not match the terminal token range"
                )
            if header["kind"] != expected_kind:
                raise ArtifactCorruptionError(
                    "artifact object kind does not match the terminal token range"
                )
        if shape_per_token is None:
            shape_per_token = array.shape[1:]
            dtype = array.dtype
        elif array.shape[1:] != shape_per_token or array.dtype != dtype:
            raise ArtifactCorruptionError("artifact key list mixes array schemas")
        next_start += array.shape[0]
        arrays.append(array)
    if (
        expected_shape_per_token is not None
        and shape_per_token != expected_shape_per_token
    ):
        raise ArtifactCorruptionError("artifact shape does not match the model")
    if expected_dtype is not None and dtype != expected_dtype:
        raise ArtifactCorruptionError("artifact dtype does not match the model")
    result = np.concatenate(arrays, axis=0)
    if expected_token_end is not None and len(result) != (
        expected_token_end - expected_token_start
    ):
        raise ArtifactCorruptionError(
            "artifact keys do not cover the terminal token range"
        )
    return result


@dataclass(frozen=True)
class _PreparedCommit:
    request_id: str
    block_end: int
    objects: list[ArtifactObject]


class RoutedExpertsRequestCore:
    """Encode immutable R3 objects and produce terminal key lists."""

    def __init__(
        self,
        store: ArtifactStore,
        source: RoutedExpertsArtifactBuffer,
    ) -> None:
        self._store = store
        self._source = source

    def _prepare_commit(self, request: ArtifactCommit) -> _PreparedCommit:
        if (
            request.block_start < 0
            or request.block_start % request.hash_block_size
            or not request.block_hashes
        ):
            raise ValueError("invalid artifact full-block commit")
        block_end = request.block_start + (
            len(request.block_hashes) * request.hash_block_size
        )
        array = self._source.read(
            request.request_id,
            request.block_start,
            block_end,
        )
        if (
            array.dtype != self._source.dtype
            or array.shape[1:] != self._source.shape_per_token
        ):
            raise RuntimeError("routed-experts capture profile changed")

        objects: list[ArtifactObject] = []
        for block_offset, block_hash in enumerate(request.block_hashes):
            source_start = request.block_start + block_offset * request.hash_block_size
            key = routed_experts_key(block_hash, request.artifact_namespace)
            local_start = source_start - request.block_start
            block_array = array[local_start : local_start + request.hash_block_size]
            objects.append(
                ArtifactObject(
                    key=key,
                    payload=encode_routed_experts_array(
                        key=key,
                        kind="block",
                        array=block_array,
                        source_token_start=source_start,
                    ),
                )
            )
        return _PreparedCommit(
            request_id=request.request_id,
            block_end=block_end,
            objects=objects,
        )

    def commit(self, requests: list[ArtifactCommit]) -> None:
        """Publish complete blocks and release their captured rows."""
        if not requests:
            return
        commits = [self._prepare_commit(request) for request in requests]
        objects = [obj for commit in commits for obj in commit.objects]
        self._store.put(objects)
        for commit in commits:
            self._source.release_through(commit.request_id, commit.block_end)

    def _put_tail(
        self,
        request: ArtifactFinalize,
        *,
        source_start: int,
        source_end: int,
    ) -> str:
        try:
            array = self._source.read(
                request.request_id,
                source_start,
                source_end,
            )
        except RuntimeError as buffer_error:
            # Frontend stop matching can finalize behind async scheduler
            # progress. In that case this partial range was released from the
            # worker buffer after its containing full block became immutable.
            block_index = source_start // request.hash_block_size
            if block_index >= len(request.block_hashes):
                raise
            block_hash = request.block_hashes[block_index]
            block_key = routed_experts_key(block_hash, request.artifact_namespace)
            payloads = self._store.get([block_key])
            if len(payloads) != 1:
                raise RuntimeError(
                    "artifact backend returned the wrong object count"
                ) from buffer_error
            block, header = decode_routed_experts_array(
                payloads[0],
                expected_key=block_key,
            )
            if (
                header["kind"] != "block"
                or header["source_token_start"] != source_start
                or block.shape[0] != request.hash_block_size
                or source_end - source_start >= request.hash_block_size
            ):
                raise ArtifactCorruptionError(
                    "artifact block cannot reconstruct the terminal tail"
                ) from buffer_error
            array = block[: source_end - source_start]
        if request.tail_block_hash is None:
            raise RuntimeError("terminal artifact is missing its partial block hash")
        key = routed_experts_key(
            request.tail_block_hash,
            request.artifact_namespace,
        )
        self._store.put(
            [
                ArtifactObject(
                    key=key,
                    payload=encode_routed_experts_array(
                        key=key,
                        kind="tail",
                        array=array,
                        source_token_start=source_start,
                    ),
                )
            ]
        )
        return key

    def finalize(self, request: ArtifactFinalize) -> list[str]:
        if request.token_end <= 0:
            raise ValueError(f"invalid artifact token range: [0, {request.token_end})")
        full_end = (
            request.token_end // request.hash_block_size * request.hash_block_size
        )
        full_block_count = full_end // request.hash_block_size
        if len(request.block_hashes) < full_block_count:
            raise RuntimeError(
                "terminal artifact is missing KV-compatible block hashes: "
                f"request={request.request_id}, expected={full_block_count}, "
                f"actual={len(request.block_hashes)}"
            )
        keys = [
            routed_experts_key(
                request.block_hashes[block_index],
                request.artifact_namespace,
            )
            for block_index in range(full_block_count)
        ]
        if full_end < request.token_end:
            keys.append(
                self._put_tail(
                    request,
                    source_start=full_end,
                    source_end=request.token_end,
                )
            )

        self._source.discard(request.request_id)
        return keys

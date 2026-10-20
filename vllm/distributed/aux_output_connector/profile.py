# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Versioned, fixed-width descriptions of indexer selections."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from functools import cached_property
from typing import Any

import numpy as np


@dataclass(frozen=True)
class IndexerSource:
    layer_id: int
    shape: tuple[int, ...]
    semantic: str
    codec: str = "int32"
    ratio: int = 1
    topk: int = 0
    consumer_layer_ids: tuple[int, ...] = ()
    trailing_count: bool = False
    selection_width: bool = False

    def __post_init__(self) -> None:
        if self.layer_id < 0 or not self.shape or min(self.shape) <= 0:
            raise ValueError("Indexer source requires a layer and a positive shape")
        if self.codec not in ("int32", "qsa_blocks", "pool_mask"):
            raise ValueError(f"Unsupported indexer codec: {self.codec}")
        if self.ratio < 1 or self.topk < 0:
            raise ValueError("Invalid indexer ratio/topk")
        if self.codec != "int32" and (
            self.topk <= 0 or self.topk % self.ratio or len(self.shape) != 1
        ):
            raise ValueError("Compact indexer codecs require divisible topk")
        if self.codec == "qsa_blocks" and self.shape != (self.topk // self.ratio + 2,):
            raise ValueError(
                "QSA block payload requires position and visibility columns"
            )
        if self.codec == "pool_mask" and (
            self.ratio > 8 or self.shape != (self.topk // self.ratio + self.ratio - 1,)
        ):
            raise ValueError("Pool payload has an invalid width/validity-mask size")

    @property
    def width(self) -> int:
        return math.prod(self.shape)


@dataclass(frozen=True)
class AuxOutputProfile:
    model_identity: str
    sources: tuple[IndexerSource, ...]
    kind: str = "indexer"
    codec_version: int = 1
    storage_dtype: str = "int32"

    def __post_init__(self) -> None:
        if self.kind != "indexer" or not self.model_identity:
            raise ValueError("Invalid indexer artifact identity")
        if self.codec_version != 1 or self.storage_dtype != "int32":
            raise ValueError("Unsupported indexer profile version/dtype")
        ids = [source.layer_id for source in self.sources]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("Indexer profile requires unique source layers")

    @property
    def shape_per_token(self) -> tuple[int, ...]:
        return (sum(source.width for source in self.sources),)

    @property
    def fixed_bytes_per_token(self) -> int:
        return self.shape_per_token[0] * 4

    def to_dict(self) -> dict[str, Any]:
        # JSON normalizes tuples so local and IPC profiles compare identically.
        return json.loads(json.dumps(asdict(self)))

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> AuxOutputProfile:
        value = dict(value)
        sources = []
        for item in value.pop("sources"):
            item = dict(item)
            item["shape"] = tuple(item["shape"])
            item["consumer_layer_ids"] = tuple(item["consumer_layer_ids"])
            sources.append(IndexerSource(**item))
        return cls(sources=tuple(sources), **value)

    @cached_property
    def profile_id(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(encoded).hexdigest()


def decode_indexer(rows: np.ndarray, profile: AuxOutputProfile) -> np.ndarray:
    """Decode source selections, preserving order, padding and count columns.

    Source-layer and consumer-layer mappings remain in the accompanying profile.
    Region selections retain their KV-group dimension.
    """
    rows = np.asarray(rows)
    if rows.dtype != np.int32 or rows.shape[1:] != profile.shape_per_token:
        raise ValueError("Indexer payload does not match its profile")
    decoded = []
    offset = 0
    for source in profile.sources:
        values = rows[:, offset : offset + source.width]
        offset += source.width
        if source.codec == "int32":
            result = values.reshape(len(rows), *source.shape)
        elif source.codec == "pool_mask":
            r = source.ratio
            groups = source.topk // r
            packed = values[:, :groups].astype(np.int64)
            pool_ids = (packed >> r) - 1
            bits = np.arange(r, dtype=np.int64)
            tokens = pool_ids[:, :, None] * r + bits
            valid = (packed[:, :, None] & (1 << bits)) != 0
            history = np.where(valid, tokens, -1).reshape(len(rows), source.topk)
            result = np.concatenate((history, values[:, groups:]), axis=1)
        else:
            r = source.ratio
            groups = source.topk // r
            blocks = values[:, :groups]
            positions, visible = values[:, groups], values[:, groups + 1]
            columns = np.arange(source.topk + r - 1)[None, :]
            count = np.minimum(visible, groups)[:, None] * r
            rank = np.minimum(columns // r, groups - 1)
            block = np.take_along_axis(
                blocks, np.broadcast_to(rank, (len(rows), rank.shape[1])), axis=1
            )
            tokens = block.astype(np.int64) * r + columns % r
            tail_start = ((positions.astype(np.int64) + 1) // r * r)[:, None]
            tail_count = ((positions + 1) % r)[:, None]
            tail_offset = columns - count
            is_tail = (
                (columns >= count) & (tail_offset < tail_count) & (tail_offset < r - 1)
            )
            tokens = np.where(columns < count, tokens, tail_start + tail_offset)
            result = np.where(((columns < count) | is_tail) & (tokens >= 0), tokens, -1)
            result = np.concatenate((result, count + tail_count), axis=1)
        decoded.append(result.astype(np.int32, copy=False))
    if len({array.shape[1:] for array in decoded}) != 1:
        raise ValueError("Heterogeneous indexer sources require packed output")
    return np.stack(decoded, axis=1)


def validate_shm_budget(
    *,
    num_blocks: int,
    scheduler_block_size: int,
    bytes_per_token: int,
    max_bytes: int | None,
) -> int:
    required = num_blocks * scheduler_block_size * bytes_per_token
    if max_bytes is not None and max_bytes < required:
        supported = max_bytes // (scheduler_block_size * bytes_per_token)
        gib = 1024**3
        raise ValueError(
            f"Required AuxOutput SHM: {required / gib:.3f} GiB; "
            f"Allowed AuxOutput SHM: {max_bytes / gib:.3f} GiB. "
            f"Current KV Blocks: {num_blocks}; Maximum Supported Blocks: {supported}. "
            "Increase aux_output_config.max_bytes, enable compact indexer encoding, "
            "or reduce KV blocks using gpu_memory_utilization or "
            "kv_cache_memory_bytes. Host staging and runtime memory are additional."
        )
    return required if max_bytes is None else max_bytes

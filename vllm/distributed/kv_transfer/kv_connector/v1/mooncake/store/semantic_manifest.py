# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fail-closed commit protocol for Mooncake semantic region objects."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from vllm.v1.core.kv_cache_utils import BlockHash

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
    ChunkedTokenDatabase,
    KeyMetadata,
    PoolKey,
    endpoint_neutral_region_metadata,
)

SEMANTIC_COMMIT_REGION = "__semantic_commit_v1__"


class MooncakeManifestStore(Protocol):
    """Subset of the Mooncake client used by the commit protocol."""

    def batch_is_exist(self, keys: list[str]) -> Sequence[int]: ...

    def batch_put_from_multi_buffers(
        self,
        keys: list[str],
        addrs: list[list[int]],
        sizes: list[list[int]],
        replicate_config: Any,
    ) -> Sequence[int]: ...


@dataclass(frozen=True)
class SemanticRegionManifest:
    """Expected semantic fields for one model shard and wire schema."""

    schema: str
    region_databases: tuple[ChunkedTokenDatabase, ...]

    def __post_init__(self) -> None:
        if not self.schema:
            raise ValueError("Semantic manifest schema must not be empty")
        if not self.region_databases:
            raise ValueError("Semantic manifest must contain at least one region")

        region_ids = [db.metadata.region_id for db in self.region_databases]
        if any(not region_id for region_id in region_ids):
            raise ValueError("Semantic manifest regions require semantic IDs")
        if len(set(region_ids)) != len(region_ids):
            raise ValueError("Semantic manifest region IDs must be unique")

        shard_identities = {
            (
                db.metadata.cache_prefix,
                db.metadata.model_name,
                db.metadata.store_namespace,
                db.metadata.tp_rank,
                db.metadata.pcp_rank,
                db.metadata.dcp_rank,
            )
            for db in self.region_databases
        }
        if len(shard_identities) != 1:
            raise ValueError("Semantic manifest regions must belong to one shard")

        for db in self.region_databases:
            if len(db.block_len) != 1 or len(db.block_stride) != 1:
                raise ValueError("Semantic manifest regions must have one field")

    @property
    def wire_entries(self) -> tuple[tuple[str, int], ...]:
        """Canonical fields; physical strides are endpoint-local details."""
        return tuple(
            sorted(
                (db.metadata.region_id, db.block_len[0]) for db in self.region_databases
            )
        )

    @property
    def wire_fingerprint(self) -> str:
        payload = json.dumps(
            (self.schema, self.wire_entries),
            separators=(",", ":"),
        ).encode()
        return hashlib.blake2b(payload, digest_size=16).hexdigest()

    @property
    def commit_metadata(self) -> KeyMetadata:
        base = self.region_databases[0].metadata
        return endpoint_neutral_region_metadata(
            base,
            f"{SEMANTIC_COMMIT_REGION}:{self.wire_fingerprint}",
        )

    def data_keys(self, chunk_hash: BlockHash) -> list[str]:
        return [db.key_for(chunk_hash) for db in self.region_databases]

    def commit_key(self, chunk_hash: BlockHash) -> str:
        return PoolKey(self.commit_metadata, chunk_hash.hex()).to_string()


class SemanticCommitProtocol:
    """Publish commits only after all fields exist and revalidate on load."""

    def __init__(
        self,
        store: MooncakeManifestStore,
        replicate_config: Any,
    ) -> None:
        self.store = store
        self.replicate_config = replicate_config

    def _all_exist(self, keys: list[str]) -> bool:
        if not keys:
            return False
        try:
            results = self.store.batch_is_exist(keys)
        except Exception:
            return False
        return len(results) == len(keys) and all(result == 1 for result in results)

    def publish(
        self,
        manifest: SemanticRegionManifest,
        chunk_hash: BlockHash,
        *,
        marker_addr: int,
        marker_size: int = 1,
    ) -> bool:
        """Publish one immutable commit after every expected region is visible."""
        if marker_size <= 0:
            raise ValueError("Semantic commit marker size must be positive")
        if not self._all_exist(manifest.data_keys(chunk_hash)):
            return False

        commit_key = manifest.commit_key(chunk_hash)
        if self._all_exist([commit_key]):
            return True
        try:
            results = self.store.batch_put_from_multi_buffers(
                [commit_key],
                [[marker_addr]],
                [[marker_size]],
                self.replicate_config,
            )
        except Exception:
            return False
        return len(results) == 1 and results[0] >= 0 and self._all_exist([commit_key])

    def is_loadable(
        self,
        manifest: SemanticRegionManifest,
        chunk_hash: BlockHash,
    ) -> bool:
        """Reject partial groups even when their commit object still exists."""
        return self._all_exist([manifest.commit_key(chunk_hash)]) and self._all_exist(
            manifest.data_keys(chunk_hash)
        )

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-side artifacts retained by the weight-cache daemon.

Artifacts are small, keyed byte strings.  They are separate from CUDA IPC
weights so a daemon can exchange them over a remote JSON control plane without
exposing GPU handles or model state.
"""

import json
from dataclasses import fields
from typing import Any

import pybase64 as base64

from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_cache.protocol import (
    MAX_ARTIFACT_SIZE,
    ArtifactCacheKey,
    dataclass_to_json,
    json_to_dataclass,
)

logger = init_logger(__name__)

MAX_CACHED_ARTIFACTS = 8
MAX_TRANSFERRED_ARTIFACT_BYTES = 1 << 24


class ArtifactStore:
    """Bounded exact-key artifact store used by one daemon rank."""

    def __init__(self, max_entries: int = MAX_CACHED_ARTIFACTS):
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        self.max_entries = max_entries
        self._artifacts: dict[ArtifactCacheKey, bytes] = {}

    def get(self, key: ArtifactCacheKey) -> bytes | None:
        return self._artifacts.get(key)

    def put(self, key: ArtifactCacheKey, data: bytes) -> None:
        if len(data) > MAX_ARTIFACT_SIZE:
            raise ValueError(f"artifact exceeds {MAX_ARTIFACT_SIZE} bytes")
        if key not in self._artifacts and len(self._artifacts) >= self.max_entries:
            self._artifacts.pop(next(iter(self._artifacts)))
        self._artifacts[key] = data

    def to_json(
        self, max_bytes: int = MAX_TRANSFERRED_ARTIFACT_BYTES
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        total = 0
        for key, data in self._artifacts.items():
            encoded = base64.b64encode(data).decode("ascii")
            if total + len(encoded) > max_bytes:
                logger.warning("Skipping artifact %r over transfer budget", key.kind)
                continue
            result.append({"key": dataclass_to_json(key), "data": encoded})
            total += len(encoded)
        return result

    def merge_json(self, payload: Any) -> int:
        if not isinstance(payload, list):
            return 0
        adopted = 0
        for item in payload:
            try:
                key = json_to_dataclass(ArtifactCacheKey, item["key"])
                if not all(isinstance(getattr(key, f.name), str) for f in fields(key)):
                    raise TypeError("artifact key fields must be strings")
                data = base64.b64decode(item["data"], validate=True)
                if len(data) > MAX_ARTIFACT_SIZE:
                    raise ValueError("artifact exceeds size limit")
                if key.kind != "flashinfer_autotune":
                    raise ValueError("unsupported artifact kind")
                if not isinstance(json.loads(data), dict):
                    raise ValueError("autotune table must be a JSON object")
                self.put(key, data)
            except Exception:
                logger.warning("Skipping malformed artifact received from peer")
                continue
            adopted += 1
        return adopted

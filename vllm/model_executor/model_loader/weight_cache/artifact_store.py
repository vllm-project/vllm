# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Storage backends for daemon warmup-cache artifacts (disk and shared stores)."""

import os
import tempfile
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Protocol

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake_store import rdma_utils
from vllm.distributed.mooncake_store import MooncakeStoreConfig, setup_mooncake_store
from vllm.logger import init_logger
from vllm.utils.import_utils import PlaceholderModule
from vllm.utils.network_utils import get_ip

try:
    from mooncake.store import MooncakeDistributedStore
except ImportError:
    mooncake_store = PlaceholderModule("mooncake.store")  # type: ignore[assignment]
    MooncakeDistributedStore = mooncake_store.placeholder_attr(
        "MooncakeDistributedStore"
    )

logger = init_logger(__name__)

_KEY_PREFIX = "vllm-warmup-cache"


def artifact_key(kind: str, config_hash: str, filename: str) -> str:
    """Canonical key for one warmup-cache artifact, valid in every store."""
    return f"{_KEY_PREFIX}/{kind}/{config_hash}/{filename}"


class ArtifactStore(Protocol):
    """One warmup-cache artifact store: bytes in and out, keyed by artifact_key."""

    name: str

    def get(self, key: str) -> bytes | None:
        """Fetch the artifact, or None on a miss."""
        ...

    def put(self, key: str, data: bytes) -> None:
        """Publish the artifact under the key."""
        ...


class DiskArtifactStore(ArtifactStore):
    """The local disk store: one directory tree per warmup-cache kind."""

    name = "disk"

    def __init__(self, kind: str, root: Path):
        self._prefix = f"{_KEY_PREFIX}/{kind}/"
        self._root = root

    def _path(self, key: str) -> Path:
        rel = key.removeprefix(self._prefix)
        parts = rel.split("/")
        if rel == key or len(parts) != 2 or any(p in ("", ".", "..") for p in parts):
            raise ValueError(f"key {key!r} is outside this store")
        return self._root / parts[0] / parts[1]

    def get(self, key: str) -> bytes | None:
        path = self._path(key)
        return path.read_bytes() if path.is_file() else None

    def put(self, key: str, data: bytes) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            dir=path.parent, suffix=".tmp", prefix=f".{path.name}."
        )
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            os.replace(tmp_path, path)
        except BaseException:
            with suppress(OSError):
                os.unlink(tmp_path)
            raise


class MooncakeArtifactStore(ArtifactStore):
    """MooncakeDistributedStore wrapper via MOONCAKE_CONFIG_PATH."""

    name = "mooncake"

    def __init__(self) -> None:
        self._store = MooncakeDistributedStore()
        config = MooncakeStoreConfig.load_from_config()
        local_hostname = rdma_utils.get_requester_local_hostname(get_ip())
        setup_mooncake_store(self._store, config, local_hostname)

    def get(self, key: str) -> bytes | None:
        """Fetch the artifact; a miss or a native error reads as empty."""
        return self._store.get(key) or None

    def put(self, key: str, data: bytes) -> None:
        """Publish the artifact; first writer wins, so publishers are idempotent."""
        if self._store.is_exist(key) == 1:
            return
        ret = self._store.put(key, data)
        if ret != 0 and self._store.is_exist(key) != 1:
            logger.warning("Mooncake artifact put %s failed (ret=%d)", key, ret)


# Shared backends for preload --warmup-cache-stores; a new backend is one entry.
STORE_BACKENDS: dict[str, Callable[[], ArtifactStore]] = {
    "mooncake": MooncakeArtifactStore,
}

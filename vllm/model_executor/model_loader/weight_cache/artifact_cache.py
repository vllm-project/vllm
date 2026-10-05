# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The weight cache daemon's host-side artifact store and its client.

Besides the weights it maps over CUDA IPC, a daemon keeps a small set of
host-side artifacts that an engine computed once and handed back, currently
the FlashInfer autotune table. They cost only host memory and are plain
bytes, so the daemon can serve one to any local process that reaches its
socket rather than only to the engine that owns its GPU.
"""

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_cache.protocol import (
    MAX_ARTIFACT_SIZE,
    ArtifactCacheKey,
    WeightCacheUnavailableError,
    connect_daemon,
    get_current_device_uuid,
    get_socket_path,
    recv_msg,
    send_msg,
)

logger = init_logger(__name__)

_TIMEOUT_S = 5.0

# A daemon serves one model configuration, so it needs room for only a handful
# of keys. Bounding the store keeps a client whose key keeps changing (e.g. one
# restarting with different engine flags) from growing it forever.
MAX_CACHED_ARTIFACTS = 8


class ArtifactStore:
    """Bounded in-memory artifact store, keyed exactly.

    The daemon treats keys as opaque and never returns bytes stored under a
    different key: an artifact is only reusable by a process whose inputs
    match the ones it was produced for, and the producer folds those into the
    key.
    """

    def __init__(self, max_entries: int = MAX_CACHED_ARTIFACTS):
        self.max_entries = max_entries
        self._artifacts: dict[ArtifactCacheKey, bytes] = {}

    def __len__(self) -> int:
        return len(self._artifacts)

    def get(self, key: ArtifactCacheKey) -> bytes | None:
        return self._artifacts.get(key)

    def put(self, key: ArtifactCacheKey, data: bytes) -> None:
        """Store the artifact, evicting the oldest key past the cap."""
        if key not in self._artifacts and len(self._artifacts) >= self.max_entries:
            self._artifacts.pop(next(iter(self._artifacts)))
        self._artifacts[key] = data


class DaemonArtifactCache:
    """Best-effort ``get``/``put`` against one weight cache daemon socket.

    Every operation degrades to a miss when the daemon is unreachable or
    rejects the request: the artifacts are caches, so losing one costs only
    the work of recomputing it, and an engine must never fail to start over
    a missing one.
    """

    def __init__(
        self,
        socket_path: str | None = None,
        socket_dir: str | None = None,
        timeout_s: float = _TIMEOUT_S,
    ):
        self.socket_path = socket_path
        self.socket_dir = socket_dir
        self.timeout_s = timeout_s

    @classmethod
    def from_vllm_config(cls, vllm_config: VllmConfig) -> "DaemonArtifactCache | None":
        """Client for the target daemon, or None when not loading from one.

        The artifacts describe the target engine as a whole, so they always
        live on the target daemon; the draft group's socket is never used.
        """
        load_config = vllm_config.load_config
        if load_config.load_format != "ipc_cache":
            return None
        extra_config = load_config.model_loader_extra_config or {}
        return cls(
            socket_path=extra_config.get("socket_path"),
            socket_dir=extra_config.get("socket_dir"),
        )

    def get(self, key: ArtifactCacheKey) -> bytes | None:
        """Fetch the artifact stored under ``key``, or None on any miss."""
        try:
            response = self._request({"cmd": "get_artifact", "key": key})
        except (WeightCacheUnavailableError, ConnectionError, OSError) as e:
            logger.info("Weight cache daemon has no %r artifact: %s", key.kind, e)
            return None
        status = response.get("status")
        if status == "miss":
            return None
        if status != "ok":
            logger.warning(
                "Weight cache daemon rejected the %r artifact request: %s",
                key.kind,
                response.get("message"),
            )
            return None
        data = response.get("data")
        if not isinstance(data, bytes):
            logger.warning(
                "Weight cache daemon returned a malformed %r artifact", key.kind
            )
            return None
        return data

    def put(self, key: ArtifactCacheKey, data: bytes) -> bool:
        """Hand the artifact to the daemon; False if it did not take it."""
        if len(data) > MAX_ARTIFACT_SIZE:
            logger.warning(
                "Not caching the %r artifact: %d bytes exceeds the %d byte limit",
                key.kind,
                len(data),
                MAX_ARTIFACT_SIZE,
            )
            return False
        try:
            response = self._request({"cmd": "put_artifact", "key": key, "data": data})
        except (WeightCacheUnavailableError, ConnectionError, OSError) as e:
            logger.warning("Cannot hand the %r artifact to the daemon: %s", key.kind, e)
            return False
        if response.get("status") != "ok":
            logger.warning(
                "Weight cache daemon refused the %r artifact: %s",
                key.kind,
                response.get("message"),
            )
            return False
        return True

    def _request(self, message: dict) -> dict:
        socket_path = self.socket_path or get_socket_path(
            get_current_device_uuid(), self.socket_dir
        )
        strict_perms = self.socket_path is None and self.socket_dir is None
        with connect_daemon(
            socket_path, self.timeout_s, strict_perms=strict_perms
        ) as conn:
            send_msg(conn, message)
            response = recv_msg(conn)
        if not isinstance(response, dict):
            raise WeightCacheUnavailableError(
                f"Weight cache daemon returned a malformed response: {response!r}"
            )
        return response

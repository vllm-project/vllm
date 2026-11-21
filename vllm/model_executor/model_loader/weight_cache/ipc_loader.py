# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Map post-processed weights from a local daemon's pluggable sharing backend."""

import contextlib
import dataclasses
import socket
import threading
import time
from collections.abc import Callable
from copy import copy
from typing import TypeVar

import torch
import torch.nn as nn

from vllm.config import ModelConfig, VllmConfig
from vllm.config.load import LoadConfig
from vllm.distributed import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.utils import (
    build_model_from_processed_weights,
    get_draft_load_config,
)
from vllm.model_executor.model_loader.weight_cache.ipc_backends import IPC_BACKENDS
from vllm.model_executor.model_loader.weight_cache.ipc_backends.base import (
    rebuild_entries,
    record_bytes,
)
from vllm.model_executor.model_loader.weight_cache.ipc_backends.serial_utils import (
    recv_msg,
    send_msg,
)
from vllm.model_executor.model_loader.weight_cache.protocol import (
    CacheConfigMismatchError,
    WeightCacheKey,
    WeightCacheState,
    WeightCacheUnavailableError,
    check_weight_cache_loader_support,
    get_current_device_uuid,
    get_socket_path,
    verify_socket_owner,
)
from vllm.model_executor.model_loader.weight_cache.utils import (
    build_weight_cache_key,
    is_draft_model_cacheable,
    record_external_weight_charge,
    record_external_weight_fallback,
)
from vllm.model_executor.model_loader.weight_cache.weight_seed.utils import (
    rebuild_tensors,
)
from vllm.tracing import instrument

logger = init_logger(__name__)

_CONNECT_TIMEOUT_S = 5.0
_STATE_TIMEOUT_S = 300.0
_STARTUP_RETRY_INTERVAL_S = 0.5

_T = TypeVar("_T")


class IpcModelLoader(BaseModelLoader):
    """Zero-copy loads a model using the weight cache daemon's sharing backend.

    The model is initialized on the meta device and every parameter/buffer is
    replaced by the daemon's post-quantized tensor, so
    process_weights_after_loading is skipped entirely. The engine shares the
    daemon's GPU memory.

    Extra config keys (via --model-loader-extra-config):

    - socket_path: explicit daemon socket path. Defaults to a per-GPU path
      derived from the physical GPU uuid and the cache role (target/draft).
    - socket_dir: directory containing the daemon sockets.
    - fallback: fall back to disk loading when fetching state fails because the
      daemon is unavailable or the fingerprints mismatch (default: True).
    - connect_timeout_s: socket connect timeout (default: 5.0).
    - state_timeout_s: timeout for the weight-transfer request (default: 300.0).

    The weights live in externally held backend allocations, so sleep mode
    (CuMemAllocator weight offloading) must not be used with this loader. If
    the daemon exits, the CUDA driver keeps the mapped weights alive until the
    engine exits, so the engine keeps serving; the loader watches the daemon
    and warns since a restarted daemon then holds a second copy of the weights
    on the GPU.
    """

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        extra_config = copy(load_config.model_loader_extra_config or {})
        self.socket_path: str | None = extra_config.pop("socket_path", None)
        self.socket_dir: str | None = extra_config.pop("socket_dir", None)
        # Internal: set by the engine when routing a speculative draft to the
        # daemon's draft group.
        self.is_draft = bool(extra_config.pop("is_draft", False))
        if self.is_draft and self.socket_path is not None:
            raise ValueError(
                "socket_path cannot be combined with the draft weight cache role; "
                "use socket_dir so the target and draft sockets are derived "
                "independently"
            )
        self.fallback: bool = extra_config.pop("fallback", True)
        self.connect_timeout_s: float = float(
            extra_config.pop("connect_timeout_s", _CONNECT_TIMEOUT_S)
        )
        self.state_timeout_s: float = float(
            extra_config.pop("state_timeout_s", _STATE_TIMEOUT_S)
        )
        if extra_config:
            raise ValueError(
                f"Unexpected extra config keys for load format "
                f"{load_config.load_format}: {sorted(extra_config)}"
            )

    def get_external_weight_memory(self, vllm_config: VllmConfig) -> int:
        total = self._daemon_memory()
        # The engine runs the draft model only on the last pipeline stage.
        if (
            is_draft_model_cacheable(vllm_config.speculative_config)
            and get_pp_group().is_last_rank
        ):
            # The draft group is queried with the same config the engine
            # would load the draft with.
            draft_config = get_draft_load_config(vllm_config)
            if draft_config.load_format == "ipc_cache":
                draft_loader = IpcModelLoader(draft_config)
                # An explicit draft_load_config without is_draft would resolve
                # back to the target socket and double-count it.
                if draft_loader.is_draft:
                    total += draft_loader._daemon_memory()
        return total

    def _daemon_memory(self) -> int:
        if not self.fallback:
            # The loader waits for the daemon at load time, so the weights
            # will be zero-copy mapped for sure; mirror that wait here since
            # returning 0 would over-grant the memory budget.
            return self._with_startup_wait(self._query_daemon_memory)
        # An unreachable daemon means a disk load, i.e. nothing external.
        try:
            return self._query_daemon_memory()
        except (WeightCacheUnavailableError, ConnectionError, OSError) as e:
            logger.warning(
                "Cannot query weight cache daemon memory (%s); "
                "assuming the weights are not externally held",
                e,
            )
            return 0

    def _query_daemon_memory(self) -> int:
        with self._connect(self.connect_timeout_s) as conn:
            send_msg(conn, {"cmd": "get_memory"})
            response = recv_msg(conn)
        if response.get("status") != "ok":
            raise WeightCacheUnavailableError(
                f"Weight cache daemon rejected the memory query: {response['message']}"
            )
        memory_bytes = int(response["memory_bytes"])
        record_external_weight_charge(self._resolve_socket_path(), memory_bytes)
        return memory_bytes

    def download_model(self, model_config: ModelConfig) -> None:
        DefaultModelLoader(self._fallback_load_config()).download_model(model_config)

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        raise NotImplementedError(
            "Standalone weight reload is not supported for load format 'ipc_cache'"
        )

    @instrument(span_name="Load model")
    def load_model(
        self, vllm_config: VllmConfig, model_config: ModelConfig, prefix: str = ""
    ) -> nn.Module:
        # An unsupported platform or role is a permanent misconfiguration
        # rather than a transient daemon outage, so it is raised even when
        # fallback is on.
        check_weight_cache_loader_support(
            vllm_config, model_config, is_draft=self.is_draft
        )
        try:
            state = self._fetch_state(vllm_config, model_config)
        except (WeightCacheUnavailableError, CacheConfigMismatchError) as e:
            if not self.fallback:
                raise
            logger.warning(
                "Weight cache unusable (%s); falling back to disk loading", e
            )
            model = self._fallback_load(vllm_config, model_config, prefix)
            record_external_weight_fallback(self._resolve_socket_path())
            return model
        model = self._build_model(vllm_config, model_config, prefix, state)
        self._watch_daemon()
        logger.info(
            "Mapped %d tensors from the weight cache daemon via zero-copy sharing",
            len(state.manifest["tensors"]),
        )
        return model

    def _build_model(
        self,
        vllm_config: VllmConfig,
        model_config: ModelConfig,
        prefix: str,
        state: WeightCacheState,
    ) -> nn.Module:
        tensors = self._rebuild_storages(state)
        return build_model_from_processed_weights(
            vllm_config,
            model_config,
            prefix,
            tensors,
            state.manifest,
            aliases=state.aliases,
            attrs=state.attrs,
            load_config=self.load_config,
        )

    def _rebuild_storages(self, state: WeightCacheState) -> dict[str, torch.Tensor]:
        """Rebuild manifest tensors from one imported buffer per storage."""
        buffers = [
            record_bytes(record, storage["nbytes"])
            for record, storage in zip(
                state.storages, state.manifest["storages"], strict=True
            )
        ]
        return rebuild_tensors(state.manifest, buffers)

    def _fetch_state(
        self,
        vllm_config: VllmConfig,
        model_config: ModelConfig,
    ) -> WeightCacheState:
        cache_key = build_weight_cache_key(
            vllm_config, model_config, is_draft=self.is_draft
        )
        return self._request_state(cache_key)

    def _with_startup_wait(self, op: Callable[[], _T]) -> _T:
        """Retry op until the daemon answers or the state timeout elapses;
        the daemon may still be loading the model when the engine starts."""
        deadline = time.monotonic() + self.state_timeout_s
        while True:
            try:
                return op()
            except (WeightCacheUnavailableError, ConnectionError, OSError) as e:
                if time.monotonic() >= deadline:
                    raise WeightCacheUnavailableError(
                        "Weight cache daemon did not become ready within "
                        f"{self.state_timeout_s:.1f}s: {e}"
                    ) from e
                logger.info_once(
                    "Waiting up to %.1fs for the weight cache daemon to start",
                    self.state_timeout_s,
                )
                time.sleep(
                    max(
                        0.0,
                        min(_STARTUP_RETRY_INTERVAL_S, deadline - time.monotonic()),
                    )
                )

    def _request_state(self, cache_key: WeightCacheKey) -> WeightCacheState:
        if self.fallback:
            return self._request_state_once(cache_key)
        return self._with_startup_wait(lambda: self._request_state_once(cache_key))

    def _request_state_once(self, cache_key: WeightCacheKey) -> WeightCacheState:
        with self._connect(self.state_timeout_s) as conn:
            send_msg(conn, {"cmd": "get_state", "cache_key": cache_key})
            response = recv_msg(conn)
        status = response.get("status")
        if status == "mismatch":
            raise CacheConfigMismatchError(
                f"WeightCacheKey mismatch on fields: {response['fields']}"
            )
        if status != "ok":
            raise WeightCacheUnavailableError(
                f"Weight cache daemon error: {response['message']}"
            )
        self._check_gpu_uuid(response["gpu_uuid"])
        backend = response.get("backend")
        if backend not in IPC_BACKENDS:
            raise CacheConfigMismatchError(
                f"weight_cache_backend mismatch: unsupported backend {backend!r}"
            )
        device_index = torch.accelerator.current_device_index()
        records = rebuild_entries(response["storages"], device_index)
        return WeightCacheState(
            manifest=response["manifest"],
            storages=records,
            aliases=response["aliases"],
            attrs=response["attrs"],
        )

    def _connect(self, timeout: float) -> socket.socket:
        socket_path = self._resolve_socket_path()
        # The auto-derived per-user directory is locked to 0700 and checked
        # strictly. When the operator explicitly configures a path they own the
        # trust decision, so only ownership/symlink safety is enforced.
        strict_perms = self.socket_path is None and self.socket_dir is None
        try:
            verify_socket_owner(socket_path, strict_perms=strict_perms)
        except OSError as e:
            raise WeightCacheUnavailableError(
                f"Weight cache socket {socket_path} is unavailable: {e}"
            ) from e
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        try:
            sock.connect(socket_path)
        except OSError as e:
            sock.close()
            raise WeightCacheUnavailableError(
                f"Cannot connect to weight cache daemon at {socket_path}: {e}"
            ) from e
        return sock

    def _resolve_socket_path(self) -> str:
        if self.socket_path is not None:
            return self.socket_path
        return get_socket_path(
            get_current_device_uuid(),
            self.socket_dir,
            is_draft=self.is_draft,
        )

    def _check_gpu_uuid(self, daemon_uuid: str) -> None:
        local_uuid = get_current_device_uuid()
        if daemon_uuid != local_uuid:
            raise CacheConfigMismatchError(
                f"Daemon GPU {daemon_uuid} != engine GPU {local_uuid}; "
                "check the socket path / GPU mapping"
            )

    def _watch_daemon(self) -> threading.Thread | None:
        """Watch for the exit of the daemon serving the zero-copy weights.

        The daemon holds the watch connection open for its lifetime, so the
        connection closes exactly when that daemon instance exits; a daemon
        restarted on the same socket does not mask the loss.
        """
        socket_path = self._resolve_socket_path()
        try:
            conn = self._connect(self.connect_timeout_s)
        except WeightCacheUnavailableError as e:
            logger.warning("Cannot watch the weight cache daemon: %s", e)
            return None
        try:
            send_msg(conn, {"cmd": "watch"})
            response = recv_msg(conn)
            if response.get("status") != "ok":
                raise WeightCacheUnavailableError(response["message"])
        except Exception as e:
            # Best effort: the weights are already mapped, so a failed watch
            # must not send load_model down the disk fallback.
            conn.close()
            logger.warning(
                "Cannot watch the weight cache daemon at %s: %s", socket_path, e
            )
            return None
        conn.settimeout(None)
        thread = threading.Thread(
            target=self._wait_for_daemon_exit,
            args=(conn, socket_path),
            daemon=True,
            name="WeightCacheDaemonWatch",
        )
        thread.start()
        return thread

    def _wait_for_daemon_exit(self, conn: socket.socket, socket_path: str) -> None:
        with conn, contextlib.suppress(OSError):
            # The daemon never writes on a watch connection; recv returns
            # b"" once it exits.
            while conn.recv(1):
                pass
        logger.warning(
            "Weight cache daemon at %s exited. This engine keeps serving: the "
            "CUDA driver keeps its mapped weights alive until the engine exits. "
            "A restarted daemon holds a second copy of the weights on this GPU, "
            "and engine restarts miss the cache until a daemon serves again.",
            socket_path,
        )

    def _fallback_load_config(self) -> LoadConfig:
        # DefaultModelLoader must not see load_format="ipc_cache" or the ipc
        # extra config keys.
        return dataclasses.replace(
            self.load_config,
            load_format="auto",
            model_loader_extra_config={},
        )

    def _fallback_load(
        self, vllm_config: VllmConfig, model_config: ModelConfig, prefix: str
    ) -> nn.Module:
        loader = DefaultModelLoader(self._fallback_load_config())
        return loader.load_model(
            vllm_config=vllm_config, model_config=model_config, prefix=prefix
        )

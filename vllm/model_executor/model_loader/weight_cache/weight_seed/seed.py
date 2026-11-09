# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The weights seedable: a daemon both serves and fetches its shard."""

import contextlib
import threading
import time
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_cache.transport import Seedable
from vllm.model_executor.model_loader.weight_cache.weight_seed.core import (
    PEER_IPC_BACKEND,
    RDMA_BACKEND,
    SeedBackend,
    WeightMover,
    acknowledge_lease,
    get_mover,
    parse_seed_backend,
)
from vllm.model_executor.model_loader.weight_cache.weight_seed.utils import (
    WeightsSeedPayload,
    build_manifest,
    validate_weights_payload,
)

if TYPE_CHECKING:
    from vllm.model_executor.model_loader.weight_cache.daemon import (
        WeightCacheDaemon,
    )

logger = init_logger(__name__)

WEIGHT_COPY_LEASE_TIMEOUT_S = 300.0


class WeightShardSeed(Seedable):
    """This daemon's weight shard: served to and fetched from peer daemons."""

    kind = "weights"

    def __init__(self, daemon: "WeightCacheDaemon"):
        # The daemon this rank serves and fetches for: the manifest, mover
        # seeds and fingerprint are all read from it.
        self._daemon = daemon
        self._movers: dict[SeedBackend, WeightMover] = {}
        self._active_copies = 0
        self._lease_condition = threading.Condition(daemon._state_lock)
        self._release_pending = False

    def _mover(self, backend: SeedBackend) -> WeightMover:
        mover = self._movers.get(backend)
        if mover is None:
            mover = get_mover(backend, device_index=self._daemon.device_index)
            self._movers[backend] = mover
        return mover

    def release(self) -> None:
        """Wait for copies, then drop source state without blocking their acks."""
        deadline = time.monotonic() + WEIGHT_COPY_LEASE_TIMEOUT_S
        with self._lease_condition:
            self._release_pending = True
            if self._active_copies:
                logger.info(
                    "Weight cache %s daemon rank %d waiting for %d "
                    "weight-cache copy acknowledgement(s)",
                    self._daemon.role,
                    self._daemon.global_rank,
                    self._active_copies,
                )
            while self._active_copies:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"Timed out waiting for {self._active_copies} "
                        "weight-cache copy acknowledgement(s); cached "
                        "weights and NIC registrations were retained"
                    )
                self._lease_condition.wait(remaining)
            # A failed release stays pending: some regions may have been
            # unregistered, so publishing the old bookkeeping is unsafe.
            for mover in self._movers.values():
                mover.release()

    def _acknowledge(self) -> dict[str, bool]:
        with self._lease_condition:
            if not self._active_copies:
                raise ValueError("Unexpected weight-cache acknowledgement")
            self._active_copies -= 1
            self._lease_condition.notify_all()
        return {"acknowledged": True}

    def validate_remote(self, request: dict[str, Any]) -> None:
        """Refuse node-local movers on the remote control plane."""
        if request.get("acknowledge"):
            return
        backend = parse_seed_backend(request.get("seed_backend", PEER_IPC_BACKEND))
        if backend == PEER_IPC_BACKEND:
            raise ValueError(
                f"Seed backend {backend!r} copies through CUDA IPC, which "
                "cannot cross hosts; mirror it over this daemon's Unix socket"
            )

    def resolve(self, request: dict[str, Any]) -> WeightsSeedPayload | dict[str, bool]:
        """Describe this rank so a peer daemon can mirror it."""
        if request.get("acknowledge"):
            return self._acknowledge()
        daemon = self._daemon
        backend = parse_seed_backend(request.get("seed_backend", PEER_IPC_BACKEND))
        with daemon._state_lock:
            if not daemon.has_weights or self._release_pending:
                raise ValueError("Weights were released")
            mover = self._mover(backend)
            tensors, kinds, aliases, attrs = daemon._seed_state()
            manifest, storages = build_manifest(tensors, kinds)
            seed = mover.prepare_seed(storages)
            with self._lease_condition:
                self._active_copies += 1
            return {
                "cache_config": asdict(daemon.cache_key),
                # Matched instead of the checkpoint fingerprint by a mirror
                # without the safetensors to hash (see seed_mismatches).
                "model": daemon.model_config.model,
                "manifest": manifest,
                "aliases": aliases,
                "attrs": attrs,
                "seed": seed,
            }

    def request_params(self, ctx: Any) -> dict[str, Any]:
        """Ask this daemon's source for the shard, over the fitting mover."""
        source = self._daemon.d2d.source
        backend = PEER_IPC_BACKEND if isinstance(source, str) else RDMA_BACKEND
        return {"seed_backend": backend}

    def validate_payload(self, payload: Any) -> WeightsSeedPayload:
        """Structural checks plus the shard fingerprint against this daemon."""
        daemon = self._daemon
        try:
            return validate_weights_payload(
                payload, daemon.cache_key, daemon.model_config.model
            )
        except Exception:
            with contextlib.suppress(Exception):
                acknowledge_lease(daemon.d2d)
            raise

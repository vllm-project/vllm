# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The weights seedable: a daemon both serves and fetches its shard."""

import contextlib
import secrets
import threading
import time
from collections import deque
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_cache.transport import Seedable
from vllm.model_executor.model_loader.weight_cache.weight_seed.core import (
    PEER_IPC_BACKEND,
    RDMA_BACKEND,
    SeedBackend,
    WeightMover,
    cancel_lease,
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
COMPLETED_LEASE_HISTORY_SIZE = 1024


class WeightShardSeed(Seedable):
    """This daemon's weight shard: served to and fetched from peer daemons."""

    kind = "weights"

    def __init__(self, daemon: "WeightCacheDaemon"):
        # The daemon this rank serves and fetches for: the manifest, mover
        # seeds and fingerprint are all read from it.
        self._daemon = daemon
        self._movers: dict[SeedBackend, WeightMover] = {}
        self._active_copy_leases: set[str] = set()
        self._completed_copy_leases: deque[str] = deque()
        self._completed_copy_lease_ids: set[str] = set()
        self._lease_condition = threading.Condition(daemon._state_lock)
        self._release_pending = False

    def _mover(self, backend: SeedBackend) -> WeightMover:
        mover = self._movers.get(backend)
        if mover is None:
            mover = get_mover(backend, device_index=self._daemon.device_index)
            self._movers[backend] = mover
        return mover

    @property
    def active_copies(self) -> int:
        with self._lease_condition:
            return len(self._active_copy_leases)

    @property
    def active_lease_ids(self) -> tuple[str, ...]:
        """Active IDs an orchestrator can explicitly cancel after target exit."""
        with self._lease_condition:
            return tuple(sorted(self._active_copy_leases))

    @property
    def release_pending(self) -> bool:
        with self._lease_condition:
            return self._release_pending

    def release(self, *, require_eplb_writer_detached: bool = False) -> None:
        """Wait for copies, then drop source state without blocking their acks."""
        deadline = time.monotonic() + WEIGHT_COPY_LEASE_TIMEOUT_S
        with self._lease_condition:
            if self._daemon.cache_key.enable_eplb:
                if (
                    require_eplb_writer_detached
                    and getattr(self._daemon, "_eplb_writer_token", None) is not None
                ):
                    raise ValueError(
                        "Cannot release EPLB weights with an active writer; "
                        "detach first"
                    )
                if self._active_copy_leases:
                    raise ValueError(
                        "Cannot release EPLB weights with active D2D copies"
                    )
            self._release_pending = True
            if self._active_copy_leases:
                logger.info(
                    "Weight cache %s daemon rank %d waiting for %d "
                    "weight-cache copy acknowledgement(s)",
                    self._daemon.role,
                    self._daemon.global_rank,
                    len(self._active_copy_leases),
                )
            while self._active_copy_leases:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"Timed out waiting for {len(self._active_copy_leases)} "
                        "weight-cache copy acknowledgement(s); cached "
                        "weights and NIC registrations were retained"
                    )
                self._lease_condition.wait(remaining)
            # A failed release stays pending: some regions may have been
            # unregistered, so publishing the old bookkeeping is unsafe.
            for mover in self._movers.values():
                mover.release()

    def _finish_lease(self, lease_id: object) -> bool:
        with self._lease_condition:
            if not isinstance(lease_id, str):
                raise ValueError("Unknown weight-cache lease ID")
            if lease_id in self._completed_copy_lease_ids:
                return False
            if lease_id not in self._active_copy_leases:
                raise ValueError("Unknown weight-cache lease ID")
            self._active_copy_leases.remove(lease_id)
            self._completed_copy_leases.append(lease_id)
            self._completed_copy_lease_ids.add(lease_id)
            if len(self._completed_copy_leases) > COMPLETED_LEASE_HISTORY_SIZE:
                evicted = self._completed_copy_leases.popleft()
                self._completed_copy_lease_ids.remove(evicted)
            daemon = self._daemon
            if (
                not self._active_copy_leases
                and daemon.cache_key.enable_eplb
                and getattr(daemon, "_eplb_writer_token", None) is None
            ):
                daemon._eplb_frozen = False
            self._lease_condition.notify_all()
            return True

    def _acknowledge(self, lease_id: object) -> dict[str, bool]:
        completed_now = self._finish_lease(lease_id)
        return {"acknowledged": True, "already_completed": not completed_now}

    def _cancel(self, lease_id: object) -> dict[str, bool]:
        completed_now = self._finish_lease(lease_id)
        return {"cancelled": True, "already_completed": not completed_now}

    def validate_remote(self, request: dict[str, Any]) -> None:
        """Refuse node-local movers on the remote control plane."""
        if (
            request.get("acknowledge")
            or request.get("cancel")
            or request.get("list_leases")
        ):
            return
        backend = parse_seed_backend(request.get("seed_backend", PEER_IPC_BACKEND))
        if backend == PEER_IPC_BACKEND:
            raise ValueError(
                f"Seed backend {backend!r} copies through CUDA IPC, which "
                "cannot cross hosts; mirror it over this daemon's Unix socket"
            )

    def resolve(self, request: dict[str, Any]) -> WeightsSeedPayload | dict[str, Any]:
        """Describe this rank so a peer daemon can mirror it."""
        if request.get("list_leases"):
            with self._lease_condition:
                return {
                    "lease_ids": list(self.active_lease_ids),
                    "active_copies": len(self._active_copy_leases),
                }
        if request.get("acknowledge"):
            return self._acknowledge(request.get("lease_id"))
        if request.get("cancel"):
            return self._cancel(request.get("lease_id"))
        if request.get("lease_protocol") != 1:
            raise ValueError(
                "Weight seeding requires lease_protocol=1 before weights are exported"
            )
        daemon = self._daemon
        backend = parse_seed_backend(request.get("seed_backend", PEER_IPC_BACKEND))
        with daemon._state_lock:
            if not daemon.has_weights or self._release_pending:
                raise ValueError("Weights were released")
            daemon.assert_eplb_seedable()
            mover = self._mover(backend)
            tensors, kinds, aliases, attrs = daemon._seed_state()
            manifest, storages = build_manifest(tensors, kinds)
            seed = mover.prepare_seed(storages)
            placement = daemon.eplb_placement
            with self._lease_condition:
                lease_id = secrets.token_urlsafe(32)
                while (
                    lease_id in self._active_copy_leases
                    or lease_id in self._completed_copy_lease_ids
                ):
                    lease_id = secrets.token_urlsafe(32)
                self._active_copy_leases.add(lease_id)
            return {
                "lease_id": lease_id,
                "cache_config": asdict(daemon.cache_key),
                # Matched instead of the checkpoint fingerprint by a mirror
                # without the safetensors to hash (see seed_mismatches).
                "model": daemon.model_config.model,
                "manifest": manifest,
                "aliases": aliases,
                "attrs": attrs,
                "placement": (
                    placement.to_json_dict() if placement is not None else None
                ),
                "seed": seed,
            }

    def request_params(self, ctx: Any) -> dict[str, Any]:
        """Ask this daemon's source for the shard, over the fitting mover."""
        source = self._daemon.d2d.source
        backend = PEER_IPC_BACKEND if isinstance(source, str) else RDMA_BACKEND
        return {"seed_backend": backend, "lease_protocol": 1}

    def validate_payload(self, payload: Any) -> WeightsSeedPayload:
        """Structural checks plus the shard fingerprint against this daemon."""
        daemon = self._daemon
        try:
            return validate_weights_payload(
                payload, daemon.cache_key, daemon.model_config.model
            )
        except Exception:
            if isinstance(payload, dict):
                with contextlib.suppress(Exception):
                    cancel_lease(daemon.d2d, payload.get("lease_id"))
            raise

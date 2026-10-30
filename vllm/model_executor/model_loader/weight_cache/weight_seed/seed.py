# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The weights seedable: one kind of state, both ends on the same daemon.

Every weight cache daemon serves its shard to peers (``resolve``) and, when
it is itself a mirror, fetches its shard from a peer (``request_params`` /
``validate_payload``); the daemon is both ends of the exchange over its
lifetime. The ``peer_cache`` model loader (:mod:`.loader`) only installs a
fetched payload into a model: it drives the mover's data plane and builds
the model around the tensors.
"""

from typing import TYPE_CHECKING, Any

from vllm.model_executor.model_loader.weight_cache.protocol import (
    dataclass_to_json,
)
from vllm.model_executor.model_loader.weight_cache.transport import Seedable
from vllm.model_executor.model_loader.weight_cache.weight_seed.core import (
    PEER_IPC_BACKEND,
    RDMA_BACKEND,
    SeedBackend,
    WeightMover,
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


class WeightShardSeed(Seedable):
    """This daemon's weight shard, seedable to and from a peer daemon.

    The payload is JSON-safe except for a ``peer_ipc`` seed: TensorEntry
    CUDA handles only ever flow over the local pickle plane, because
    ``validate_remote`` refuses node-local movers on the remote JSON plane.

    Owns the mover registry: each mover is created on first use and held
    for the daemon's lifetime so source-side state (e.g. Mooncake's
    registered regions) serves every later mirror.
    """

    kind = "weights"

    def __init__(self, daemon: "WeightCacheDaemon"):
        # The daemon this rank serves and fetches for: the manifest, mover
        # seeds and fingerprint are all read from it.
        self._daemon = daemon
        self._movers: dict[SeedBackend, WeightMover] = {}

    def _mover(self, backend: SeedBackend) -> WeightMover:
        mover = self._movers.get(backend)
        if mover is None:
            mover = get_mover(backend, device_index=self._daemon.device_index)
            self._movers[backend] = mover
        return mover

    def release(self) -> None:
        """Drop every mover's source-side state for released weights."""
        for mover in self._movers.values():
            mover.release()

    def validate_remote(self, request: dict[str, Any]) -> None:
        """Refuse node-local movers on the remote control plane."""
        backend = parse_seed_backend(request.get("seed_backend", PEER_IPC_BACKEND))
        if self._mover(backend).is_node_local:
            raise ValueError(
                f"Seed backend {backend!r} copies through CUDA IPC, which "
                "cannot cross hosts; mirror it over this daemon's Unix socket"
            )

    def resolve(self, request: dict[str, Any]) -> WeightsSeedPayload:
        """Describe this rank so a peer daemon can mirror it."""
        daemon = self._daemon
        backend = parse_seed_backend(request.get("seed_backend", PEER_IPC_BACKEND))
        with daemon._state_lock:
            if not daemon.has_weights:
                raise ValueError("Weights were released")
            mover = self._mover(backend)
            tensors, kinds, aliases, attrs = daemon._seed_state()
            manifest, storages = build_manifest(tensors, kinds)
            seed = mover.prepare_seed(storages)
            return {
                "cache_config": dataclass_to_json(daemon.cache_key),
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
        return validate_weights_payload(
            payload, daemon.cache_key, daemon.model_config.model
        )

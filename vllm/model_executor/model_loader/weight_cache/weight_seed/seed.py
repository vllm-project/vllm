# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The weights seedable: both ends of weight-shard seeding.

The source side (:class:`WeightShardSeed`) is served by every weight cache
daemon over the D2D transport: it resolves a request into a storage manifest
plus a mover seed. The target side (:class:`WeightShardTarget`) is driven by
the ``peer_cache`` model loader (:mod:`.loader`) through
``SeedTransport.fetch`` + ``install``: it validates the fingerprint and
its ``install`` drives the mover's data plane, stashing the filled
tensors for the loader to install into its model.
"""

from typing import TYPE_CHECKING, Any, cast

import torch

from vllm.model_executor.model_loader.weight_cache.protocol import (
    TensorKind,
    WeightCacheKey,
    dataclass_to_json,
    json_to_dataclass,
)
from vllm.model_executor.model_loader.weight_cache.transport import Seedable
from vllm.model_executor.model_loader.weight_cache.weight_seed.core import (
    PEER_IPC_BACKEND,
    SeedBackend,
    WeightMover,
    get_mover,
    parse_seed_backend,
)
from vllm.model_executor.model_loader.weight_cache.weight_seed.utils import (
    WeightsSeedPayload,
    build_manifest,
    seed_mismatches,
)

if TYPE_CHECKING:
    from vllm.model_executor.model_loader.weight_cache.daemon import (
        WeightCacheDaemon,
    )


class WeightShardSeed(Seedable):
    """The source side of weight-shard seeding to a peer daemon.

    The payload is JSON-safe except for a ``peer_ipc`` seed: TensorEntry
    CUDA handles only ever flow over the local pickle plane, because
    ``validate_remote`` refuses node-local movers on the remote JSON plane.

    Owns the mover registry: each mover is created on first use and held
    for the daemon's lifetime so source-side state (e.g. Mooncake's
    registered regions) serves every later mirror.
    """

    kind = "weights"

    def __init__(self, daemon: "WeightCacheDaemon"):
        # The daemon this rank serves: the manifest, mover seeds and
        # fingerprint are all read from it at resolve time.
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


class WeightShardTarget(Seedable):
    """The target side of weight-shard seeding, driven by the peer loader.

    A mirror has no local path — its host may not have the checkpoint — so
    the loader uses the strict ``fetch`` and every failure is fatal.
    """

    kind = "weights"

    def __init__(
        self,
        backend: SeedBackend,
        expected_key: WeightCacheKey,
        model: str,
        device: torch.device,
    ):
        self._backend = backend
        self._expected_key = expected_key
        self._model = model
        self._device = device
        self.tensors: dict[str, torch.Tensor] = {}
        self.kinds: dict[str, TensorKind] = {}
        self.aliases: dict[str, str] = {}
        self.attrs: dict[str, bool] = {}

    def request_params(self, ctx: Any) -> dict[str, Any]:
        return {"seed_backend": self._backend}

    def validate_payload(self, payload: Any) -> WeightsSeedPayload:
        validated = _validate_payload(payload)
        source_key = json_to_dataclass(WeightCacheKey, validated.get("cache_config"))
        mismatched = seed_mismatches(
            self._expected_key, source_key, self._model, validated.get("model")
        )
        if mismatched:
            hint = (
                f"; no safetensors under {self._model} to fingerprint here, "
                "so the mirror needs the source's --model path or the "
                "checkpoint files"
                if mismatched == ["checkpoint"]
                and self._expected_key.checkpoint == self._model
                else ""
            )
            raise ValueError(
                f"Seed daemon cache differs on fields {mismatched}; refusing "
                f"to mirror a different weight shard{hint}"
            )
        return validated

    def install(self, ctx: Any, payload: WeightsSeedPayload) -> None:
        """Copy the source's storages with the payload's mover."""
        manifest = payload["manifest"]
        mover = get_mover(parse_seed_backend(payload["seed"]["backend"]))
        self.tensors = mover.fill(manifest, payload["seed"], self._device)
        self.kinds = {
            name: "param" if view["is_param"] else "buffer"
            for name, view in manifest["tensors"].items()
        }
        self.aliases = dict(payload["aliases"])
        self.attrs = dict(payload["attrs"])


def _validate_payload(payload: Any) -> WeightsSeedPayload:
    """Structural checks on the source daemon's weights seed payload."""
    if not isinstance(payload, dict):
        raise ValueError("Seed daemon returned a malformed payload")
    model = payload.get("model")
    if model is not None and not isinstance(model, str):
        raise ValueError("Seed daemon returned a malformed model name")
    manifest = payload.get("manifest")
    if (
        not isinstance(manifest, dict)
        or not isinstance(manifest.get("storages"), list)
        or not isinstance(manifest.get("tensors"), dict)
    ):
        raise ValueError("Seed daemon returned a malformed manifest")
    seed = payload.get("seed")
    if not isinstance(seed, dict) or not isinstance(seed.get("backend"), str):
        raise ValueError("Seed daemon returned a malformed transfer seed")
    for key in ("aliases", "attrs"):
        if payload.get(key) is not None and not isinstance(payload[key], dict):
            raise ValueError(f"Seed daemon returned malformed {key}")
    return cast(WeightsSeedPayload, payload)

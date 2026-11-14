# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compiled runtime artifact warmup cache."""

import hashlib
import time
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_cache.artifact_store import (
    STORE_BACKENDS,
    ArtifactStore,
    artifact_key,
)
from vllm.model_executor.model_loader.weight_cache.warmup_cache.base import WarmupCache
from vllm.model_executor.model_loader.weight_cache.warmup_cache.common import (
    BUNDLE_LIMIT,
    _bundle,
    _cache_roots,
    _install,
    _manifest,
)

if TYPE_CHECKING:
    from vllm.model_executor.model_loader.weight_cache.daemon import WeightCacheDaemon
    from vllm.model_executor.model_loader.weight_cache.seed import SeedTransport

logger = init_logger(__name__)


class CompileArtifactWarmupCache(WarmupCache):
    """Compiled runtime cache seeded as part of daemon warmup."""

    kind = "compile_cache"

    def __init__(self, store_names: Iterable[str] = ()) -> None:
        self.roots = _cache_roots()
        self._store_names = tuple(store_names)
        self._store_sources: list[ArtifactStore] | None = None
        self._payload: dict[str, Any] | None = None

    @property
    def store_sources(self) -> list[ArtifactStore]:
        if self._store_sources is None:
            self._store_sources = [STORE_BACKENDS[name]() for name in self._store_names]
        return self._store_sources

    def adopt_before_load(
        self, daemon: "WeightCacheDaemon", transport: "SeedTransport"
    ) -> None:
        if daemon.seed_config.source is None or not self.store_sources:
            return
        deadline = time.monotonic() + 300.0
        while time.monotonic() < deadline:
            response = transport.request(
                {"kind": self.kind, **self.request_params(daemon)}
            )
            if response is not None and response.get("status") == "ok":
                try:
                    payload = self.validate_payload(response.get("payload"))
                    self.install(daemon, payload)
                except Exception as error:
                    logger.warning(
                        "Cannot install remote compile cache; using local loading: %s",
                        error,
                    )
                    return
                logger.info("Adopted remote compile cache before model loading")
                return
            if response is not None and response.get("status") == "miss":
                return
            if response is not None and response.get("status") == "error":
                logger.warning(
                    "Source does not provide a compile cache; using local loading"
                )
                return
            time.sleep(1.0)
        logger.warning(
            "Timed out waiting for the source compile cache; using local loading"
        )

    def warmup(self, daemon: "WeightCacheDaemon", transport: "SeedTransport") -> None:
        # Compiler outputs are created by the preceding FlashInfer warmup. A
        # target already fetched its bundle before model construction.
        if daemon.seed_config.listen is not None:
            self.prepare()

    def prepare(self) -> None:
        self._payload = None
        manifest = _manifest(self.roots)
        stores = self.store_sources
        if not manifest["entries"] or not stores:
            return
        bundle = _bundle(self.roots, manifest)
        digest = hashlib.sha256(bundle).hexdigest()
        key = artifact_key(self.kind, digest, "bundle.tar")
        for store in stores:
            store.put(key, bundle)
        self._payload = {
            "manifest": manifest,
            "key": key,
            "bundle_size": len(bundle),
            "sha256": digest,
        }

    def request_params(self, ctx: Any) -> dict[str, Any]:
        return {"version": 1}

    def resolve(self, request: dict[str, Any]) -> dict[str, Any] | None:
        if request.get("version") != 1:
            return None
        return self._payload

    def validate_payload(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("compile cache payload must be an object")
        manifest = payload.get("manifest")
        if not isinstance(manifest, dict) or manifest.get("version") != 1:
            raise ValueError("malformed compiled cache manifest")
        if manifest["total_size"] > BUNDLE_LIMIT:
            raise ValueError("compiled cache manifest is too large")
        size = payload.get("bundle_size")
        key, digest = payload.get("key"), payload.get("sha256")
        if isinstance(size, int) and size > BUNDLE_LIMIT:
            raise ValueError("compiled cache bundle is too large")
        if (
            not manifest["entries"]
            or not isinstance(size, int)
            or size <= 0
            or not isinstance(digest, str)
            or key != artifact_key(self.kind, digest, "bundle.tar")
            or sum(entry["size"] for entry in manifest["entries"])
            != manifest["total_size"]
        ):
            raise ValueError("invalid compile cache descriptor")
        return {
            "manifest": manifest,
            "key": key,
            "bundle_size": size,
            "sha256": digest,
        }

    def install(self, ctx: Any, payload: dict[str, Any]) -> None:
        for store in self.store_sources:
            bundle = store.get(payload["key"])
            if bundle is None:
                continue
            if len(bundle) != payload["bundle_size"]:
                raise ValueError("compiled cache bundle size mismatch")
            if hashlib.sha256(bundle).hexdigest() != payload["sha256"]:
                raise ValueError("compiled cache bundle checksum mismatch")
            _install(bundle, payload["manifest"], self.roots)
            return
        raise RuntimeError("compiled cache bundle is missing from every shared store")

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compiled runtime artifact warmup cache."""

import hashlib
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from vllm.model_executor.model_loader.weight_cache.artifact_store import (
    STORE_BACKENDS,
    ArtifactStore,
    artifact_key,
)
from vllm.model_executor.model_loader.weight_cache.seed import Seedable

from .common import (
    BUNDLE_LIMIT,
    CacheManifest,
    CacheRoot,
    _bundle,
    _install,
    _manifest,
    _safe_path,
)

if TYPE_CHECKING:
    from vllm.model_executor.model_loader.weight_cache.daemon import WeightCacheDaemon
    from vllm.model_executor.model_loader.weight_cache.seed import SeedTransport


class CompileArtifactWarmupCache(Seedable):
    """Compiled runtime cache seeded as part of daemon warmup."""

    kind = "compile_cache"

    def __init__(
        self,
        roots: tuple[CacheRoot, ...],
        store_names: Iterable[str] = (),
    ) -> None:
        self.roots = roots
        self._store_names = tuple(store_names)
        self._store_sources: list[ArtifactStore] | None = None
        self._manifest: CacheManifest | None = None
        self._bundle: bytes | None = None
        self._digest: str | None = None
        self._key: str | None = None

    @property
    def store_sources(self) -> list[ArtifactStore]:
        if self._store_sources is None:
            self._store_sources = [STORE_BACKENDS[name]() for name in self._store_names]
        return self._store_sources

    def warmup(self, daemon: "WeightCacheDaemon", transport: "SeedTransport") -> None:
        # Compiler outputs are created by the preceding FlashInfer warmup. A
        # target already fetched its bundle before model construction.
        if daemon.seed_config.listen is not None:
            self.prepare()

    def prepare(self) -> None:
        self._manifest = _manifest(self.roots)
        self._bundle = _bundle(self.roots, self._manifest)
        self._digest = hashlib.sha256(self._bundle).hexdigest()
        self._key = artifact_key(self.kind, self._digest, "bundle.tar")

    def request_params(self, ctx: Any) -> dict[str, Any]:
        return {"version": 1}

    def resolve(self, request: dict[str, Any]) -> dict[str, Any] | None:
        if (
            request.get("version") != 1
            or self._manifest is None
            or self._bundle is None
            or self._digest is None
            or self._key is None
            or not self._manifest.entries
        ):
            return None
        stores = self.store_sources
        if not stores:
            return None
        for store in stores:
            store.put(self._key, self._bundle)
        return {
            "manifest": self._manifest.as_dict(),
            "key": self._key,
            "bundle_size": len(self._bundle),
            "sha256": self._digest,
        }

    def validate_payload(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("compile cache payload must be an object")
        raw_manifest = payload.get("manifest")
        if not isinstance(raw_manifest, dict) or raw_manifest.get("version") != 1:
            raise ValueError("malformed compiled cache manifest")
        raw_entries = raw_manifest.get("entries")
        if not isinstance(raw_entries, list):
            raise ValueError("malformed compiled cache entries")
        entries = tuple(raw_entries)
        if any(not isinstance(entry, dict) for entry in entries):
            raise ValueError("malformed compiled cache entry")
        total = raw_manifest.get("total_size")
        if not isinstance(total, int) or total < 0:
            raise ValueError("malformed compiled cache size")
        manifest = CacheManifest(entries, total)
        size = payload.get("bundle_size")
        key, digest = payload.get("key"), payload.get("sha256")
        if (
            not isinstance(size, int)
            or size < 0
            or not isinstance(key, str)
            or not isinstance(digest, str)
        ):
            raise ValueError("malformed compile cache descriptor")
        if total > BUNDLE_LIMIT:
            raise ValueError("compiled cache manifest is too large")
        if size > BUNDLE_LIMIT:
            raise ValueError("compiled cache bundle is too large")
        if manifest.entries and size == 0:
            raise ValueError("invalid compile cache bundle size")
        if not manifest.entries and size != 0:
            raise ValueError("empty compile cache has a non-empty bundle")
        if len(digest) != 64:
            raise ValueError("malformed compiled cache bundle checksum")
        try:
            int(digest, 16)
        except ValueError as error:
            raise ValueError("malformed compiled cache bundle checksum") from error
        if key != artifact_key(self.kind, digest, "bundle.tar"):
            raise ValueError("compiled cache bundle key mismatch")
        root_names = {root.name for root in self.roots}
        for entry in entries:
            if (
                not isinstance(entry.get("root"), str)
                or not isinstance(entry.get("path"), str)
                or not isinstance(entry.get("size"), int)
                or entry["size"] < 0
                or not isinstance(entry.get("sha256"), str)
                or not isinstance(entry.get("mtime_ns"), int)
            ):
                raise ValueError("malformed compiled cache entry")
            if entry["root"] not in root_names:
                raise ValueError("unknown compiled cache root")
            _safe_path(entry["path"])
            if len(entry["sha256"]) != 64:
                raise ValueError("malformed compiled cache checksum")
            try:
                int(entry["sha256"], 16)
            except ValueError as error:
                raise ValueError("malformed compiled cache checksum") from error
        if sum(entry["size"] for entry in entries) != total:
            raise ValueError("compiled cache manifest size mismatch")
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

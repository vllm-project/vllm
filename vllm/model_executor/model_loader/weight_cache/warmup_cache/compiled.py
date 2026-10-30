# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compiled runtime artifact warmup cache."""

import ctypes
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_cache.seed import Seedable
from vllm.utils.import_utils import has_mooncake

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

logger = init_logger(__name__)


class CompileArtifactWarmupCache(Seedable):
    """Compiled runtime cache seeded as part of daemon warmup."""

    kind = "compile_cache"

    def __init__(
        self,
        roots: tuple[CacheRoot, ...],
        protocol: str = "rdma",
        require_mooncake: bool = False,
    ):
        if protocol not in ("tcp", "rdma"):
            raise ValueError(f"unsupported Mooncake protocol: {protocol!r}")
        if require_mooncake and not has_mooncake():
            raise ImportError(
                "compiled-cache D2D seeding requires the Mooncake TransferEngine"
            )
        self.roots = roots
        self.protocol = protocol
        self.require_mooncake = require_mooncake
        self._manifest: CacheManifest | None = None
        self._bundle: bytes | None = None
        self._engine: Any = None
        self._buffer: Any = None
        self._source_address: int | None = None
        self._source_session: str | None = None

    def warmup(self, daemon: "WeightCacheDaemon", transport: "SeedTransport") -> None:
        # Compiler outputs are created by the preceding FlashInfer warmup. A
        # target already fetched its bundle before model construction.
        if daemon.seed_config.listen is not None:
            self.prepare()

    def prepare(self) -> None:
        self._manifest = _manifest(self.roots)
        self._bundle = _bundle(self.roots, self._manifest)

    def request_params(self, ctx: Any) -> dict[str, Any]:
        return {"version": 1}

    def resolve(self, request: dict[str, Any]) -> dict[str, Any] | None:
        if request.get("version") != 1 or self._manifest is None:
            return None
        if self._bundle is None:
            raise RuntimeError("compile cache bundle is not ready")
        if not self._manifest.entries:
            return None
        try:
            descriptor = self._ensure_source()
        except (ImportError, RuntimeError) as error:
            logger.warning(
                "Mooncake is unavailable for compile-cache seeding: %s", error
            )
            return None
        return {
            "manifest": self._manifest.as_dict(),
            "bundle_size": len(self._bundle),
            "transport": "mooncake",
            "protocol": self.protocol,
            **descriptor,
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
        address, session = payload.get("address"), payload.get("session")
        if (
            not isinstance(size, int)
            or size < 0
            or payload.get("transport") != "mooncake"
            or not isinstance(address, int)
            or not isinstance(session, str)
        ):
            raise ValueError("malformed compile cache transfer descriptor")
        if total > BUNDLE_LIMIT:
            raise ValueError("compiled cache manifest is too large")
        if size > BUNDLE_LIMIT:
            raise ValueError("compiled cache bundle is too large")
        if manifest.entries and size == 0:
            raise ValueError("invalid compile cache bundle size")
        if not manifest.entries and size != 0:
            raise ValueError("empty compile cache has a non-empty bundle")
        protocol = payload.get("protocol")
        if protocol not in ("tcp", "rdma") or protocol != self.protocol:
            raise ValueError("unsupported Mooncake protocol")
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
            "bundle_size": size,
            "protocol": payload.get("protocol", "rdma"),
            "address": address,
            "session": session,
        }

    def install(self, ctx: Any, payload: dict[str, Any]) -> None:
        bundle = self._read_mooncake(payload)
        _install(bundle, payload["manifest"], self.roots)

    def _ensure_source(self) -> dict[str, Any]:
        if self._engine is None or self._buffer is None:
            try:
                from mooncake.engine import TransferEngine

                from vllm.utils.network_utils import get_ip
            except ImportError as error:
                raise RuntimeError("compile-cache seeding requires Mooncake") from error
            if self._bundle is None:
                raise RuntimeError("compile cache bundle is not ready")
            engine = TransferEngine()
            if engine.initialize(get_ip(), "P2PHANDSHAKE", self.protocol, "") != 0:
                raise RuntimeError("Mooncake TransferEngine initialization failed")
            buffer = ctypes.create_string_buffer(self._bundle)
            address = ctypes.addressof(buffer)
            if engine.register_memory(address, len(self._bundle)) != 0:
                raise RuntimeError("Mooncake failed to register compile-cache buffer")
            self._engine, self._buffer = engine, buffer
            self._source_address = address
            self._source_session = f"{get_ip()}:{engine.get_rpc_port()}"
        assert self._source_address is not None
        assert self._source_session is not None
        return {"address": self._source_address, "session": self._source_session}

    def _read_mooncake(self, payload: dict[str, Any]) -> bytes:
        try:
            from mooncake.engine import TransferEngine

            from vllm.utils.network_utils import get_ip
        except ImportError as error:
            raise RuntimeError("compile-cache seeding requires Mooncake") from error
        size = payload["bundle_size"]
        engine = TransferEngine()
        if engine.initialize(get_ip(), "P2PHANDSHAKE", payload["protocol"], "") != 0:
            raise RuntimeError("Mooncake TransferEngine initialization failed")
        buffer = ctypes.create_string_buffer(size)
        address = ctypes.addressof(buffer)
        if engine.register_memory(address, size) != 0:
            raise RuntimeError("Mooncake failed to register compile-cache buffer")
        try:
            status = engine.batch_transfer_sync_read(
                payload["session"], [address], [payload["address"]], [size]
            )
            if status != 0:
                raise RuntimeError(f"Mooncake transfer failed with status {status}")
            return buffer.raw[:size]
        finally:
            engine.unregister_memory(address)

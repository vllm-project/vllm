# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Daemon warmup caches, seeded daemon-to-daemon over the shared transport.

Each warmup-cache object (currently only :class:`FlashInferWarmupCache`)
drives its own warmup pass on the daemon; the seeding flow itself
(:class:`seed.SeedTransport`) is shared with future kinds such as weight
shards. A target daemon installs its rank-matched cache before tuning, so
both the daemon's own warmup and the later engine start skip re-tuning. The
disk layout is the index: a request names the cache by the same key the
layout uses (kind, config hash, dp rank, world rank) and the source resolves
it to files under the cache root. There is no separate artifact store or
registry.

The FlashInfer payload is a ``{filename: parsed JSON}`` map rather than a
single document because upcoming kinds (Triton tuned configs) consist of
several JSON files per cache.

TODO: install valid files individually instead of falling back to local
warmup when any file of a multi-file kind fails validation.
"""

import ctypes
import gc
import hashlib
import io
import json
import os
import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Protocol

import regex as re
import torch

from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_cache.seed import Seedable
from vllm.model_executor.model_loader.weight_cache.utils import (
    build_warmup_runner,
    is_draft_model_cacheable,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer

if TYPE_CHECKING:
    from vllm.model_executor.model_loader.weight_cache.daemon import (
        WeightCacheDaemon,
    )
    from vllm.model_executor.model_loader.weight_cache.seed import (
        SeedTransport,
    )
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

logger = init_logger(__name__)

_BUNDLE_LIMIT = 8 << 30


@dataclass(frozen=True)
class CacheRoot:
    name: str
    path: Path


@dataclass(frozen=True)
class CacheManifest:
    entries: tuple[dict[str, Any], ...]
    total_size: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "total_size": self.total_size,
            "entries": list(self.entries),
        }


def _safe_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe cache path: {value!r}")
    return path


def _cache_roots() -> tuple[CacheRoot, ...]:
    roots = []
    for name, variable in (
        ("flashinfer-cubins", "FLASHINFER_CUBIN_DIR"),
        ("flashinfer-workspace", "FLASHINFER_WORKSPACE_BASE"),
    ):
        value = os.environ.get(variable)
        if value:
            roots.append(CacheRoot(name, Path(value).expanduser()))
    return tuple(roots)


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest(roots: tuple[CacheRoot, ...]) -> CacheManifest:
    entries: list[dict[str, Any]] = []
    total = 0
    for root in roots:
        if not root.path.is_dir():
            continue
        for path in sorted(root.path.rglob("*")):
            if (
                not path.is_file()
                or path.is_symlink()
                or path.name.endswith((".lock", ".tmp"))
            ):
                continue
            relative = path.relative_to(root.path).as_posix()
            _safe_path(relative)
            stat = path.stat()
            total += stat.st_size
            if total > _BUNDLE_LIMIT:
                raise ValueError("compiled cache is too large")
            entries.append(
                {
                    "root": root.name,
                    "path": relative,
                    "size": stat.st_size,
                    "sha256": _digest(path),
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
    return CacheManifest(tuple(entries), total)


def _bundle(roots: tuple[CacheRoot, ...], manifest: CacheManifest) -> bytes:
    by_name = {root.name: root for root in roots}
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:") as archive:
        for entry in manifest.entries:
            root = by_name[entry["root"]]
            path = root.path / _safe_path(entry["path"])
            if not path.is_file() or path.stat().st_size != entry["size"]:
                raise ValueError(f"compiled cache changed: {path}")
            info = tarfile.TarInfo(f"{entry['root']}/{entry['path']}")
            info.size, info.mode = entry["size"], 0o600
            info.mtime = entry["mtime_ns"] / 1e9
            with path.open("rb") as stream:
                archive.addfile(info, stream)
    return output.getvalue()


def _install(
    bundle: bytes, manifest: CacheManifest, roots: tuple[CacheRoot, ...]
) -> None:
    by_name = {root.name: root for root in roots}
    expected = {f"{entry['root']}/{entry['path']}": entry for entry in manifest.entries}
    staged: list[tuple[Path, Path, dict[str, Any]]] = []
    try:
        with tarfile.open(fileobj=io.BytesIO(bundle), mode="r:") as archive:
            members = archive.getmembers()
            if len(members) != len(expected):
                raise ValueError("compiled cache bundle entry count mismatch")
            for member in members:
                entry = expected.get(member.name)
                if entry is None or not member.isfile():
                    raise ValueError(f"unexpected compiled cache member: {member.name}")
                root_name, relative = member.name.split("/", 1)
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ValueError("cannot read compiled cache member")
                data = extracted.read()
                if (
                    len(data) != entry["size"]
                    or hashlib.sha256(data).hexdigest() != entry["sha256"]
                ):
                    raise ValueError(f"compiled cache checksum mismatch: {member.name}")
                destination = by_name[root_name].path / _safe_path(relative)
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_name(destination.name + ".tmp")
                temporary.write_bytes(data)
                staged.append((temporary, destination, entry))
        for temporary, destination, entry in staged:
            os.replace(temporary, destination)
            os.utime(destination, ns=(entry["mtime_ns"], entry["mtime_ns"]))
    except Exception:
        for temporary, _, _ in staged:
            temporary.unlink(missing_ok=True)
        raise


_CONFIG_HASH_RE = re.compile(r"[0-9a-f]{64}")


class WarmupCache(Seedable, Protocol):
    """One daemon warmup cache: its warmup pass and its cache's semantics.

    The warmup layer; the D2D exchange lives in seed.py (SeedTransport).
    """

    def warmup(self, daemon: "WeightCacheDaemon", transport: "SeedTransport") -> None:
        """Run this rank's warmup pass against the cached model."""
        ...


class FlashInferWarmupCache(WarmupCache):
    """FlashInfer's per-rank autotune table (a single JSON file)."""

    kind = "flashinfer_autotune"

    def warmup(self, daemon: "WeightCacheDaemon", transport: "SeedTransport") -> None:
        vllm_config = daemon.vllm_config
        # Mirrors the gate in the engine's kernel warmup.
        if (
            vllm_config.kernel_config.enable_flashinfer_autotune is False
            or not has_flashinfer()
            or not current_platform.has_device_capability(90)
            or is_draft_model_cacheable(vllm_config.speculative_config)
        ):
            return
        from vllm.model_executor.warmup.kernel_warmup import flashinfer_autotune

        try:
            current_platform.update_block_size_for_backend(vllm_config)
            runner = build_warmup_runner(
                vllm_config,
                daemon.local_rank,
                is_draft=daemon.is_draft,
                model_config=daemon.model_config,
            )
            assert daemon.model is not None, "warmup ran before load_model"
            runner.load_model(model=daemon.model)
            transport.adopt(self, runner)
            # The V2 runner requires skip_attn when no KV cache exists
            flashinfer_autotune(runner, skip_attn=vllm_config.use_v2_model_runner)
            logger.info(
                "Weight cache %s daemon rank %d tuned FlashInfer; the tuned "
                "configs are in the on-disk autotune cache",
                daemon.role,
                daemon.global_rank,
            )
        finally:
            # Free the runner and the dummy-run activations; the daemon's
            # weights are untouched.
            gc.collect()
            torch.accelerator.empty_cache()

    def request_params(self, runner: "GPUModelRunner") -> dict[str, Any]:
        from vllm.distributed.parallel_state import get_world_group
        from vllm.model_executor.warmup.flashinfer_autotune_cache import (
            flashinfer_autotune_cache_hash,
        )

        return {
            "config_hash": flashinfer_autotune_cache_hash(runner),
            "dp_rank": runner.vllm_config.parallel_config.data_parallel_rank,
            "rank": get_world_group().rank_in_group,
        }

    def resolve(self, request: dict[str, Any]) -> dict[str, Any] | None:
        from vllm.model_executor.warmup.flashinfer_autotune_cache import (
            flashinfer_autotune_filename,
            flashinfer_autotune_root,
        )

        config_hash = request.get("config_hash")
        if not isinstance(config_hash, str) or not _CONFIG_HASH_RE.fullmatch(
            config_hash
        ):
            raise ValueError("invalid config_hash")

        # Garbage ranks simply name a file that does not exist (a miss); the
        # hex check above is what keeps the request inside the cache root.
        filename = flashinfer_autotune_filename(request["dp_rank"], request["rank"])
        path = flashinfer_autotune_root() / config_hash / filename
        if not path.is_file():
            return None
        return {filename: json.loads(path.read_bytes())}

    def validate_payload(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict) or len(payload) != 1:
            raise ValueError("expected exactly one autotune table")
        ((filename, table),) = payload.items()
        if not isinstance(table, dict):
            raise ValueError("autotune table must be a JSON object")
        if not filename.startswith("autotune_configs_"):
            raise ValueError(f"unexpected file {filename!r}")
        # Reject tables flashinfer would refuse to load: a hard _metadata
        # conflict would also never be overwritten, poisoning this cache
        # path into re-tuning on every start.
        saved_meta = table.get("_metadata")
        if saved_meta is None:
            # Pre-metadata flashinfer file; load_configs accepts these.
            return payload
        try:
            from flashinfer.autotuner.autotuner import (
                _classify_metadata_mismatches,
                _collect_metadata,
            )
        except ImportError:
            logger.warning(
                "flashinfer metadata classifier unavailable; "
                "accepting the remote autotune table unchecked"
            )
            return payload
        hard, soft = _classify_metadata_mismatches(saved_meta, _collect_metadata())
        mismatches = {**hard, **soft}
        if mismatches:
            details = ", ".join(
                f"{key}: remote={old} vs local={new}"
                for key, (old, new) in mismatches.items()
            )
            raise ValueError(f"autotune table environment mismatch: {details}")
        return payload

    def install(self, runner: "GPUModelRunner", payload: dict[str, Any]) -> None:
        from vllm.model_executor.warmup.flashinfer_autotune_cache import (
            resolve_rank_flashinfer_autotune_file,
            write_flashinfer_autotune_cache,
        )

        (data,) = payload.values()
        cache_path = resolve_rank_flashinfer_autotune_file(runner)
        write_flashinfer_autotune_cache(cache_path, json.dumps(data).encode())
        logger.info("Installed remote FlashInfer autotune table: %s", cache_path)


class CompileArtifactWarmupCache(WarmupCache):
    """Compiled runtime cache seeded as part of daemon warmup."""

    kind = "compile_cache"

    def __init__(self, roots: tuple[CacheRoot, ...], protocol: str = "rdma"):
        if protocol not in ("tcp", "rdma"):
            raise ValueError(f"unsupported Mooncake protocol: {protocol!r}")
        self.roots = roots
        self.protocol = protocol
        self._manifest: CacheManifest | None = None
        self._bundle: bytes | None = None
        self._engine: Any = None
        self._buffer: Any = None

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
        if manifest.entries and size == 0:
            raise ValueError("invalid compile cache bundle size")
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

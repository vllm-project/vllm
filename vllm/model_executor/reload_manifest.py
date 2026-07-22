# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capture-time manifest for module-level (global) tensor storage.

The reload arena covers storage a layer owns. Some graph-visible tensors
have no owning layer at all: they live in module-level caches and
registries, e.g. ``_g_workspace: dict[torch.device, torch.Tensor]`` in the
tokenspeed MLA backend, or the Triton sampler's buffer caches. No walk
rooted at the model reaches them, and no copy-back protects them -- but a
captured graph bakes their addresses just the same, and a cache that grows
or is repopulated rebinds them silently.

This module records those storages when graphs are captured and verifies
them at reload commit. Two independent signals, because neither alone is
sufficient:

``expired``
    The recorded storage was freed. Any graph holding that address is
    dangling. Detected with a weak reference, so it needs no path and no
    owner -- it works even for a tensor nothing can reach anymore.

``moved``
    The path still resolves, but to different storage, while the old
    storage is still alive (typically retained by a capture artifact).
    This is the silent-stale-read case: replay keeps using the previous
    values and nothing crashes. Measured live on H200 for the Machete
    act-order permutation -- 88/88 tensors rebound with 0/88 storages
    freed -- which is why an expiry-only check is not enough.

Module-level state is enumerated by scanning loaded modules rather than by
dataflow tracing. That keeps paths stable and re-resolvable (so ``moved``
is computable, which a ``TorchDispatchMode`` recorder's positional paths
are not) and keeps the production path free of op interception. Discovering
storage that this scan cannot see is a CI concern, not a serving one.
"""

import sys
from dataclasses import dataclass, field

import torch
from torch.multiprocessing.reductions import StorageWeakRef

from vllm.logger import init_logger

logger = init_logger(__name__)

__all__ = [
    "GlobalStorageManifest",
    "ManifestReport",
    "collect_module_level_tensors",
    "record_global_storage",
    "check_global_storage",
    "reset_global_storage",
]

# Only vLLM's own modules are scanned. Third-party module state is out of
# reach for a fix anyway, and scanning all of sys.modules is both slow and
# noisy.
DEFAULT_MODULE_PREFIXES = (
    "vllm.model_executor",
    "vllm.attention",
    "vllm.v1.attention",
    "vllm.v1.sample",
    "vllm.v1.worker",
    "vllm.distributed",
)

# This package's own bookkeeping would otherwise be reported as findings.
_SELF_EXCLUDED = (
    "vllm.model_executor.reload_manifest",
    "vllm.model_executor.reload_arena",
    "vllm.model_executor.model_loader.reload",
)


@dataclass(frozen=True)
class _Slot:
    ref: StorageWeakRef
    data_ptr: int
    shape: tuple[int, ...]
    dtype: torch.dtype


@dataclass
class ManifestReport:
    """Outcome of checking the manifest after a reload."""

    expired: list[str] = field(default_factory=list)
    moved: list[str] = field(default_factory=list)
    # Path no longer resolves but its storage is alive: something else holds
    # it, so no captured address dangles. Informational, not a violation.
    vanished: list[str] = field(default_factory=list)
    checked: int = 0

    @property
    def is_clean(self) -> bool:
        return not self.expired and not self.moved

    def format(self, max_items: int = 10) -> str:
        lines = [
            f"global storage: checked={self.checked} "
            f"expired={len(self.expired)} moved={len(self.moved)} "
            f"vanished={len(self.vanished)}"
        ]
        for label, paths in (("expired (captured address freed)", self.expired),
                             ("moved (stale read risk)", self.moved)):
            if paths:
                lines.append(f"  {label}:")
                lines.extend(f"    {p}" for p in paths[:max_items])
                if len(paths) > max_items:
                    lines.append(f"    ... and {len(paths) - max_items} more")
        return "\n".join(lines)


def _walk(value, path: str, depth: int, out: dict[str, torch.Tensor],
          require_cuda: bool) -> None:
    if depth < 0:
        return
    if isinstance(value, torch.Tensor):
        if not require_cuda or value.is_cuda:
            out[path] = value
        return
    if isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            _walk(item, f"{path}[{i}]", depth - 1, out, require_cuda)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            # Keys must render stably or the path is not re-resolvable
            # across the reload; skip anything that cannot.
            try:
                rendered = repr(key)
            except Exception:
                continue
            _walk(item, f"{path}[{rendered}]", depth - 1, out, require_cuda)


def collect_module_level_tensors(
    prefixes: tuple[str, ...] = DEFAULT_MODULE_PREFIXES,
    max_depth: int = 3,
    require_cuda: bool = True,
) -> dict[str, torch.Tensor]:
    """Tensors held in module-level state, keyed by a re-resolvable path.

    Args:
        require_cuda: only device tensors can have an address baked into a
            graph. Tests set this False to exercise the logic on CPU.
    """
    out: dict[str, torch.Tensor] = {}
    # Snapshot the module table: importing on another thread would
    # otherwise mutate it mid-iteration.
    for mod_name, module in list(sys.modules.items()):
        if module is None or not mod_name.startswith(prefixes):
            continue
        if mod_name.startswith(_SELF_EXCLUDED):
            continue
        try:
            members = list(vars(module).items())
        except Exception:
            continue
        for attr, value in members:
            if attr.startswith("__"):
                continue
            # Only module-level *state*, not re-exported classes/functions
            # (those carry their own module's tensors, found under that
            # module's own name).
            if isinstance(value, type) or callable(value):
                continue
            _walk(value, f"{mod_name}.{attr}", max_depth, out, require_cuda)
    return out


class GlobalStorageManifest:
    """Records module-level storage at capture; verifies it after reload."""

    def __init__(self) -> None:
        self._slots: dict[str, _Slot] = {}
        self._prefixes = DEFAULT_MODULE_PREFIXES
        self._max_depth = 3
        self._require_cuda = True

    def __len__(self) -> int:
        return len(self._slots)

    def paths(self) -> list[str]:
        return sorted(self._slots)

    def record(
        self,
        prefixes: tuple[str, ...] = DEFAULT_MODULE_PREFIXES,
        max_depth: int = 3,
        require_cuda: bool = True,
    ) -> int:
        self._prefixes = prefixes
        self._max_depth = max_depth
        self._require_cuda = require_cuda
        self._slots = {
            path: _Slot(
                ref=StorageWeakRef(t.untyped_storage()),
                data_ptr=t.data_ptr(),
                shape=tuple(t.shape),
                dtype=t.dtype,
            )
            for path, t in collect_module_level_tensors(
                prefixes, max_depth, require_cuda).items()
        }
        return len(self._slots)

    def check(self) -> ManifestReport:
        report = ManifestReport(checked=len(self._slots))
        if not self._slots:
            return report

        current = collect_module_level_tensors(
            self._prefixes, self._max_depth, self._require_cuda)

        for path, slot in self._slots.items():
            if slot.ref.expired():
                # Freed: any captured address is dangling regardless of
                # whether the path still resolves.
                report.expired.append(
                    f"{path} (was {slot.shape} {slot.dtype})")
                continue
            now = current.get(path)
            if now is None:
                report.vanished.append(path)
            elif now.data_ptr() != slot.data_ptr:
                report.moved.append(
                    f"{path} ({hex(slot.data_ptr)} -> {hex(now.data_ptr())})")
        return report


# Process-scoped: module-level storage is process-scoped too. Holds only
# weak references and integers, so it pins nothing alive.
_MANIFEST: GlobalStorageManifest | None = None


def record_global_storage(**kwargs) -> int:
    """Record module-level storage as it stands at graph-capture time."""
    global _MANIFEST
    _MANIFEST = GlobalStorageManifest()
    count = _MANIFEST.record(**kwargs)
    logger.debug("Recorded %d module-level tensor storages for reload "
                 "verification", count)
    return count


def check_global_storage() -> ManifestReport | None:
    """Verify recorded storage. None if nothing was ever recorded."""
    if _MANIFEST is None:
        return None
    return _MANIFEST.check()


def reset_global_storage() -> None:
    global _MANIFEST
    _MANIFEST = None

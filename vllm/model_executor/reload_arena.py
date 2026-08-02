# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Layer-owned stable storage for runtime tensors that must survive reload.

Every confirmed category-1 failure in RFC #48312 has the same structure: a
tensor whose device address is baked into a captured CUDA graph is owned by a
TRANSIENT object -- a kernel, experts, or quant-config object that
``process_weights_after_loading`` rebuilds from scratch on every run. The
graph outlives the rebuild; the storage does not.

The arena inverts the ownership: storage belongs to the layer (an
``nn.Module`` that survives reload), and transient objects merely borrow it
by slot name. Rebuild the kernel object as often as you like -- asking the
arena for the same slot returns the same storage, so every address a graph
captured stays valid, and every derived value lands back in the storage the
graph reads.

Three acquisition idioms cover the reproduced failure shapes:

``get_or_alloc``
    For workspaces and scratch (Marlin ``workspace``, CUTLASS
    ``ab_strides*``, MoE permute scratch): the caller wants storage of a
    given spec and does not care about prior contents (or wants them
    zeroed). First call allocates; later calls return the SAME tensor.
    This also covers lazily-allocated buffers -- the first-forward
    allocation goes through the arena, so it survives the next reload even
    though no reload machinery ever sees it.

``put``
    For derived values computed by post-load processing (MLA ``W_UV``,
    fp32 sinks copies): the caller has just recomputed the value and wants
    it published at a stable address. First call adopts a private
    contiguous copy; later calls ``copy_`` into that same storage. This
    gives storage identity (category 1) and value refresh (category 2)
    from one call, because PWAL re-runs recompute the value and ``put``
    lands it in place.

``get_or_create_object``
    For runtime wrappers that own graph-visible storage internally: the first
    call constructs the object and later PWAL rebuilds borrow the same object.

The arena is also the declaration registry: ``snapshot``/``verify`` give the
reload commit gate an exact inventory of graph-visible runtime storage, with
no reflective walking (closures, lazy allocations, and config-embedded
tensors all evade walks -- each escape was reproduced live on H200).
"""

import contextlib
from collections.abc import Callable, Iterator
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum
from typing import TypeVar, cast

import torch
from torch import nn

from vllm.logger import init_logger

logger = init_logger(__name__)

T = TypeVar("T")

__all__ = [
    "InitPolicy",
    "SlotViolation",
    "ReloadArena",
    "get_reload_arena",
    "peek_reload_arena",
    "arena_scope",
    "current_arena",
    "snapshot_model_arenas",
    "verify_model_arenas",
]


class InitPolicy(Enum):
    """Contents policy when ``get_or_alloc`` returns an existing slot."""

    EMPTY = "empty"  # first alloc uninitialized; re-acquire keeps contents
    ZERO = "zero"  # zeroed on first alloc AND on every re-acquire
    PRESERVE = "preserve"  # first alloc uninitialized; re-acquire keeps


@dataclass(frozen=True)
class SlotIdentity:
    data_ptr: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    device: str


@dataclass(frozen=True)
class ObjectIdentity:
    value: object


ArenaIdentity = SlotIdentity | ObjectIdentity


@dataclass(frozen=True)
class SlotViolation:
    slot: str
    kind: str  # "moved" | "gone" | "respecified"
    detail: str

    def __str__(self) -> str:
        return f"[{self.kind}] {self.slot}: {self.detail}"


def _identity(t: torch.Tensor) -> SlotIdentity:
    return SlotIdentity(
        data_ptr=t.data_ptr(),
        shape=tuple(t.shape),
        stride=tuple(t.stride()),
        dtype=t.dtype,
        device=str(t.device),
    )


def _canonical_device(device) -> torch.device:
    """Resolve an unindexed accelerator device to the one torch would
    actually allocate on.

    Callers commonly pass a config-level ``"cuda"``/``"xpu"`` while the
    tensor a slot holds reports ``cuda:0``; comparing them literally makes
    a re-acquire after reload fail spuriously (observed live on CUDA: the
    mismatch aborted the first post-reload forward mid-capture).

    Resolution goes through torch's per-type device module rather than
    ``current_platform``, for two reasons. The argument being canonicalized
    is a specific device, not necessarily the platform vLLM is running on
    (a CPU slot on a CUDA platform is legitimate), so the current platform
    is not the right question to ask. And ``torch.get_device_module``
    covers any backend registered with torch, including out-of-tree ones,
    which the platform enum does not. This mirrors what vLLM's own
    platform code does per device type, e.g. XpuPlatform's
    ``device.index if device.index is not None else torch.xpu.current_device()``.

    CPU is deliberately excluded: CPU tensors report a bare ``cpu`` with no
    index, so canonicalizing to ``cpu:0`` would introduce the very mismatch
    this exists to prevent.
    """
    device = torch.device(device)
    if device.index is not None or device.type == "cpu":
        return device

    get_device_module = getattr(torch, "get_device_module", None)
    if get_device_module is None:
        return device
    try:
        module = get_device_module(device.type)
        if not (module.is_available() and hasattr(module, "current_device")):
            return device
        index = module.current_device()
    except Exception:
        # Unknown/unavailable backend: leave the device as given rather
        # than guessing. Worst case the caller sees a spec mismatch, which
        # is loud and safe.
        return device
    if not isinstance(index, int):
        return device
    return torch.device(device.type, index)


class ReloadArena:
    """Stable tensor storage and runtime object slots owned by one layer.

    Not an ``nn.Module``: slot tensors must not appear as parameters or
    buffers (they are not checkpoint state and must not be touched by
    load/copy-back machinery). The arena itself IS their reload contract.
    """

    def __init__(self, owner_name: str = "") -> None:
        self._owner_name = owner_name
        self._slots: dict[str, torch.Tensor] = {}
        self._object_slots: dict[str, object] = {}

    def __contains__(self, slot: str) -> bool:
        return slot in self._slots or slot in self._object_slots

    def __len__(self) -> int:
        return len(self._slots) + len(self._object_slots)

    def slots(self) -> dict[str, torch.Tensor]:
        return dict(self._slots)

    def objects(self) -> dict[str, object]:
        return dict(self._object_slots)

    def _check_spec(self, slot: str, existing: torch.Tensor,
                    shape: tuple[int, ...], dtype: torch.dtype,
                    device: torch.device) -> None:
        if (tuple(existing.shape) != tuple(shape)
                or existing.dtype != dtype
                or existing.device != _canonical_device(device)):
            raise ValueError(
                f"ReloadArena[{self._owner_name}] slot '{slot}' re-acquired "
                f"with an incompatible spec: have "
                f"{tuple(existing.shape)}/{existing.dtype}/{existing.device}, "
                f"asked {tuple(shape)}/{dtype}/{device}. A shape-changing "
                "reload is not an in-place update; it requires a cold "
                "restart. Refusing to silently replace storage a captured "
                "graph may hold.")

    def get_or_alloc(
        self,
        slot: str,
        shape,
        dtype: torch.dtype,
        device,
        *,
        init: InitPolicy = InitPolicy.EMPTY,
    ) -> torch.Tensor:
        """Return the slot's tensor, allocating on first use.

        Later calls with the same slot return the SAME tensor (same
        storage), regardless of which object asks. Spec mismatch raises --
        never silently reallocates.
        """
        shape = tuple(shape)
        device = _canonical_device(device)
        if slot in self._object_slots:
            raise ValueError(
                f"ReloadArena[{self._owner_name}] slot '{slot}' is already "
                "used by a persistent object"
            )
        existing = self._slots.get(slot)
        if existing is not None:
            self._check_spec(slot, existing, shape, dtype, device)
            if init is InitPolicy.ZERO:
                existing.zero_()
            return existing

        if init is InitPolicy.ZERO:
            t = torch.zeros(shape, dtype=dtype, device=device)
        else:
            t = torch.empty(shape, dtype=dtype, device=device)
        self._slots[slot] = t
        return t

    def put(self, slot: str, value: torch.Tensor) -> torch.Tensor:
        """Publish a freshly-computed derived value at a stable address.

        First call adopts a private contiguous copy of ``value``; later
        calls copy the new value into that same storage and return it.
        Callers must rebind their attribute to the RETURNED tensor, not to
        ``value``.
        """
        if slot in self._object_slots:
            raise ValueError(
                f"ReloadArena[{self._owner_name}] slot '{slot}' is already "
                "used by a persistent object"
            )
        existing = self._slots.get(slot)
        if existing is None:
            stable = value.detach().contiguous().clone()
            self._slots[slot] = stable
            return stable
        self._check_spec(slot, existing, tuple(value.shape), value.dtype,
                         value.device)
        # detach: value may be a view of a live parameter
        existing.copy_(value.detach())
        return existing

    def get_or_create_object(
        self,
        slot: str,
        factory: Callable[[], T],
    ) -> T:
        """Return a layer-owned runtime object, creating it once."""
        if slot in self._slots:
            raise ValueError(
                f"ReloadArena[{self._owner_name}] slot '{slot}' is already "
                "used by a tensor"
            )
        if slot in self._object_slots:
            return cast(T, self._object_slots[slot])

        value = factory()
        self._object_slots[slot] = value
        return value

    def snapshot(self) -> dict[str, ArenaIdentity]:
        snapshot: dict[str, ArenaIdentity] = {
            name: _identity(t) for name, t in self._slots.items()
        }
        snapshot.update(
            {
                name: ObjectIdentity(value)
                for name, value in self._object_slots.items()
            }
        )
        return snapshot

    def verify(self, snap: dict[str, ArenaIdentity]) -> list[SlotViolation]:
        """Compare current slots against a snapshot.

        New slots since the snapshot are fine (first forward after a cold
        start legitimately adds lazy slots). A snapshotted slot that moved,
        vanished, or changed layout is a violation: some address consumer
        (a captured graph) may still hold the old identity.
        """
        out: list[SlotViolation] = []
        for name, ident in snap.items():
            if isinstance(ident, ObjectIdentity):
                if name not in self._object_slots:
                    out.append(
                        SlotViolation(name, "gone", "persistent object vanished")
                    )
                elif self._object_slots[name] is not ident.value:
                    out.append(
                        SlotViolation(
                            name,
                            "moved",
                            "persistent object identity changed",
                        )
                    )
                continue

            cur = self._slots.get(name)
            if cur is None:
                out.append(
                    SlotViolation(name, "gone",
                                  f"slot vanished (was {ident.shape} "
                                  f"{ident.dtype} @ {hex(ident.data_ptr)})"))
                continue
            cur_ident = _identity(cur)
            if cur_ident.data_ptr != ident.data_ptr:
                out.append(
                    SlotViolation(
                        name, "moved",
                        f"{hex(ident.data_ptr)} -> {hex(cur_ident.data_ptr)}"))
            elif (cur_ident.shape != ident.shape
                  or cur_ident.stride != ident.stride
                  or cur_ident.dtype != ident.dtype):
                out.append(
                    SlotViolation(
                        name, "respecified",
                        f"{ident.shape}/{ident.dtype} -> "
                        f"{cur_ident.shape}/{cur_ident.dtype}"))
        return out


_ARENA_ATTR = "_reload_arena"


def get_reload_arena(module: nn.Module) -> ReloadArena:
    """The module's arena, created on first use.

    Stored as a plain attribute: the arena's tensors must stay out of
    ``_parameters``/``_buffers`` (see ``ReloadArena`` docstring).
    """
    arena = getattr(module, _ARENA_ATTR, None)
    if arena is None:
        arena = ReloadArena(owner_name=type(module).__name__)
        setattr(module, _ARENA_ATTR, arena)
    return arena


def peek_reload_arena(module: nn.Module) -> "ReloadArena | None":
    return getattr(module, _ARENA_ATTR, None)


# Ambient arena for construction chains that never see the layer.
#
# Kernel/experts objects are built deep inside make_*_moe_kernel() calls whose
# signatures do not carry the layer. Rather than threading a parameter through
# every factory, the PWAL site (which HAS the layer) opens a scope:
#
#     with arena_scope(get_reload_arena(layer)):
#         self.moe_kernel = make_fp8_moe_kernel(...)
#
# and constructors resolve current_arena(). Outside any scope current_arena()
# is None and callers fall back to plain allocation -- unit tests and
# non-reloadable uses are unaffected. Constructors that allocate lazily (first
# forward) must CAPTURE the arena at construction time, not resolve it at
# forward time (no scope is open during forward).
_current_arena: ContextVar["ReloadArena | None"] = ContextVar(
    "vllm_reload_arena", default=None)


@contextlib.contextmanager
def arena_scope(arena: ReloadArena) -> Iterator[ReloadArena]:
    token = _current_arena.set(arena)
    try:
        yield arena
    finally:
        _current_arena.reset(token)


def current_arena() -> "ReloadArena | None":
    return _current_arena.get()


def snapshot_model_arenas(
        model: nn.Module) -> dict[str, dict[str, ArenaIdentity]]:
    out: dict[str, dict[str, ArenaIdentity]] = {}
    for name, module in model.named_modules():
        arena = peek_reload_arena(module)
        if arena is not None and len(arena):
            out[name] = arena.snapshot()
    return out


def verify_model_arenas(
        model: nn.Module,
        snaps: dict[str, dict[str, ArenaIdentity]]) -> list[str]:
    """Flat, human-readable violation list across the whole model."""
    problems: list[str] = []
    modules = dict(model.named_modules())
    for mod_name, snap in snaps.items():
        module = modules.get(mod_name)
        arena = peek_reload_arena(module) if module is not None else None
        if arena is None:
            problems.append(f"{mod_name}: arena vanished across reload")
            continue
        for v in arena.verify(snap):
            problems.append(f"{mod_name}: {v}")
    return problems

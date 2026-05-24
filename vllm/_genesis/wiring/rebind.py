# SPDX-License-Identifier: Apache-2.0
"""Attribute rebind framework — the live monkey-patch layer.

Use this for patches where the target is a function or method that can be
cleanly replaced without touching the surrounding file body.

Example
-------
    from vllm._genesis.wiring import AttributeRebinder
    from vllm._genesis.kernels.router_softmax import router_softmax
    import vllm.model_executor.layers.fused_moe as fm

    rebinder = AttributeRebinder(
        patch_name="P31 router fp32 softmax",
        target_module=fm,
        target_attr="softmax_gating",
        replacement=router_softmax,
    )
    rebinder.apply()
    # Later, for tests:
    assert rebinder.is_applied()
    rebinder.revert()  # restores original

Why not just `setattr(mod, name, fn)` directly?
------------------------------------------------
The rebinder:
  - Stores the original reference so tests can revert + rebind.
  - Verifies the target attribute exists before rebinding (catches typos
    + upstream renames).
  - Checks that the target is not already our kernel (idempotency).
  - Registers itself in `WiringRegistry` so `apply_all` can report what
    rebinds are live.

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

log = logging.getLogger("genesis.wiring.rebind")


class WiringRegistry:
    """Process-wide registry of applied rebinds.

    Thread/fork safety: each worker process has its own registry (vLLM uses
    spawn for workers → fresh sys.modules → fresh registry).
    """
    _entries: list["AttributeRebinder"] = []

    @classmethod
    def register(cls, rebinder: "AttributeRebinder") -> None:
        cls._entries.append(rebinder)

    @classmethod
    def all(cls) -> list["AttributeRebinder"]:
        return list(cls._entries)

    @classmethod
    def clear_for_tests(cls) -> None:
        """Revert all + clear. Tests only."""
        for e in reversed(cls._entries):
            try:
                e.revert()
            except Exception:
                pass
        cls._entries.clear()

    @classmethod
    def summary(cls) -> dict:
        applied = sum(1 for e in cls._entries if e.is_applied())
        pending = sum(1 for e in cls._entries if not e.is_applied())
        return {
            "total": len(cls._entries),
            "applied": applied,
            "pending_or_reverted": pending,
            "entries": [
                {
                    "patch_name": e.patch_name,
                    "target": f"{e._module_name}.{e.target_attr}",
                    "applied": e.is_applied(),
                }
                for e in cls._entries
            ],
        }


@dataclass
class AttributeRebinder:
    """Swaps `target_module.target_attr` to `replacement`, reversibly."""
    patch_name: str
    target_module: Any  # typically an imported module; may be a class too
    target_attr: str
    replacement: Callable
    _original: Optional[Callable] = field(default=None, init=False, repr=False)
    _applied: bool = field(default=False, init=False)
    _module_name: str = field(default="", init=False)

    def __post_init__(self) -> None:
        self._module_name = getattr(
            self.target_module, "__name__", repr(self.target_module),
        )

    def apply(self) -> bool:
        """Apply the rebind. Returns True if newly applied, False if already/error."""
        if self._applied:
            return False

        if not hasattr(self.target_module, self.target_attr):
            log.warning(
                "[%s] target %s.%s not found — rebind skipped",
                self.patch_name, self._module_name, self.target_attr,
            )
            return False

        current = getattr(self.target_module, self.target_attr)

        # Idempotency: already our function?
        if current is self.replacement:
            log.debug(
                "[%s] %s.%s is already our kernel — idempotent",
                self.patch_name, self._module_name, self.target_attr,
            )
            self._original = current  # no-op, but record something
            self._applied = True
            WiringRegistry.register(self)
            return False

        self._original = current
        try:
            setattr(self.target_module, self.target_attr, self.replacement)
        except Exception as e:
            log.warning(
                "[%s] setattr failed on %s.%s: %s",
                self.patch_name, self._module_name, self.target_attr, e,
            )
            self._original = None
            return False

        self._applied = True
        WiringRegistry.register(self)
        log.info(
            "[%s] rebound %s.%s to %s",
            self.patch_name, self._module_name, self.target_attr,
            getattr(self.replacement, "__qualname__", repr(self.replacement)),
        )
        return True

    def is_applied(self) -> bool:
        """True if the live binding currently points to our replacement."""
        if not self._applied:
            return False
        if not hasattr(self.target_module, self.target_attr):
            return False
        return getattr(self.target_module, self.target_attr) is self.replacement

    def revert(self) -> bool:
        """Restore the original binding. Returns True on successful revert."""
        if not self._applied:
            return False
        if self._original is None:
            return False
        try:
            setattr(self.target_module, self.target_attr, self._original)
        except Exception as e:
            log.warning(
                "[%s] revert failed on %s.%s: %s",
                self.patch_name, self._module_name, self.target_attr, e,
            )
            return False
        self._applied = False
        return True

    def assert_applied(self) -> None:
        """Raise AssertionError if not applied. For post-register verification."""
        if not self.is_applied():
            raise AssertionError(
                f"[{self.patch_name}] {self._module_name}.{self.target_attr} "
                f"is NOT bound to Genesis replacement"
            )

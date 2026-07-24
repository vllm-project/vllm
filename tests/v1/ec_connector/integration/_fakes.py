# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""In-process NIXL stand-in for ec_connector integration tests.

The real `NixlWrapper` (UCX/RDMA-backed) is the right thing to test against
when the host has NIXL installed — see `test_scheduler_real_stack.py`. This
module covers the case where we want to exercise the *integrated* scheduler
+ worker + ZMQ + region stack on any host, without depending on the NIXL
package or its UCX runtime.

Two `FakeNixlWrapper` instances created in the same process share a
module-level agent registry (`_AGENTS`). When the consumer issues a READ
against the producer, the bytes are copied via `ctypes.memmove` from the
producer's mmap region into the consumer's — no real RDMA, no UCX, no
network — but the same address-based interface the scheduler uses against
the real wrapper. The transfer also delivers its `notif_msg` to the
producer (the data owner), which the producer drains via `get_new_notifs`.

What this fake faithfully reproduces:
    * agent name → wrapper resolution (so peer-pair tests work without
      a transport)
    * per-block descriptor lists (the scheduler's `build_block_descs`
      output flows through unchanged)
    * `make_prepped_xfer` / `transfer` / `check_xfer_state` lifecycle for
      READ (and WRITE), with the correct copy direction
    * completion notifications delivered to the data owner
    * cross-mmap byte motion identical to what a real transfer would land

What it does *not* reproduce:
    * async progression (transfer completes synchronously inside
      `transfer()` — the scheduler's poll loop still runs as written, it
      just always sees DONE on the first poll)
    * UCX/transport-level errors
    * remote-agent metadata exchange (we encode the agent name in the
      "metadata" so the other side can look us up)
"""

from __future__ import annotations

import ctypes
import threading
from typing import Any

# Module-level registry of agent_name → wrapper. Tests should call
# ``reset_fake_nixl_universe`` between cases to keep state clean.
_AGENTS: dict[str, FakeNixlWrapper] = {}
_AGENTS_LOCK = threading.Lock()


def reset_fake_nixl_universe() -> None:
    """Clear the agent registry. Call between tests."""
    with _AGENTS_LOCK:
        _AGENTS.clear()


class _FakeXferHandle:
    """One pending transfer: the per-block ``(src_addr, dst_addr, size)``
    plan, the state flag the scheduler polls via ``check_xfer_state``, and
    the completion notification to deliver to ``notify_agent`` (the data
    owner) once the copy lands."""

    __slots__ = ("plan", "state", "released", "notif_msg", "notify_agent")

    def __init__(
        self,
        plan: list[tuple[int, int, int]],
        notif_msg: bytes,
        notify_agent: str,
    ) -> None:
        self.plan = plan
        self.state: str = "PROC"
        self.released: bool = False
        self.notif_msg = notif_msg
        self.notify_agent = notify_agent


class FakeNixlWrapper:
    """Drop-in stand-in for `NixlWrapper` (the symbol the scheduler imports
    from `vllm.distributed.nixl_utils`). Patch it into the scheduler module
    via ``unittest.mock.patch`` before instantiating the scheduler."""

    def __init__(self, agent_name: str, _config: Any = None) -> None:
        self.name: str = agent_name
        self._region_base: int | None = None
        self._region_size: int | None = None
        # Completion notifications delivered to this agent (it is the data
        # owner for some remote reader); drained by get_new_notifs.
        self._notifs: list[bytes] = []
        with _AGENTS_LOCK:
            _AGENTS[agent_name] = self

    # ── identity ─────────────────────────────────────────────────────────────

    def get_agent_metadata(self) -> bytes:
        # Real NIXL ships an opaque UCX endpoint blob; the fake just needs
        # something `add_remote_agent` can decode back to a wrapper.
        return f"FAKE_NIXL:{self.name}".encode()

    def add_remote_agent(self, metadata: bytes) -> str:
        prefix = b"FAKE_NIXL:"
        assert metadata.startswith(prefix), (
            f"FakeNixlWrapper: unrecognized metadata {metadata!r}"
        )
        name = metadata[len(prefix) :].decode()
        with _AGENTS_LOCK:
            assert name in _AGENTS, (
                f"FakeNixlWrapper: peer {name!r} not in universe; both peers "
                f"must be constructed before the first WRITE"
            )
        return name

    def remove_remote_agent(self, _name: str) -> None:
        # Bookkeeping-only in the fake; the real implementation tears down
        # UCX endpoints. Tests shouldn't notice.
        pass

    # ── memory registration ─────────────────────────────────────────────────

    def get_reg_descs(self, descs: list[tuple], _mem_type: str) -> list[tuple]:
        return list(descs)

    def register_memory(self, reg_descs: list[tuple], *, backends=None) -> None:
        assert len(reg_descs) == 1, (
            "FakeNixlWrapper expects a single registered region per agent"
        )
        base, size, _devid, _name = reg_descs[0]
        self._region_base = base
        self._region_size = size

    def deregister_memory(self, _reg_descs) -> None:
        pass

    # ── transfer descriptor lists ────────────────────────────────────────────

    def get_xfer_descs(self, block_descs: list[tuple], _mem_type: str) -> list[tuple]:
        return list(block_descs)

    def prep_xfer_dlist(
        self, agent_or_init: str, xfer_descs: list[tuple]
    ) -> tuple[str, str, list[tuple]]:
        # The dlist captures which agent's descs these are; make_prepped_xfer
        # uses the descs as the source-of-truth for per-block addresses.
        return ("dlist", agent_or_init, xfer_descs)

    # ── xfer post / poll / release ──────────────────────────────────────────

    def make_prepped_xfer(
        self,
        op: str,
        local_handle: tuple,
        local_indices: list[int],
        remote_handle: tuple,
        remote_indices: list[int],
        *,
        notif_msg: bytes = b"",
    ) -> _FakeXferHandle:
        assert op in ("READ", "WRITE"), (
            f"FakeNixlWrapper supports READ/WRITE, got {op!r}"
        )
        _, local_kind, local_dlist = local_handle
        _, remote_agent, remote_dlist = remote_handle
        assert local_kind == "NIXL_INIT_AGENT", (
            f"local dlist must be NIXL_INIT_AGENT, got {local_kind!r}"
        )
        if len(local_indices) != len(remote_indices):
            raise ValueError(
                f"FakeNixlWrapper: block count mismatch "
                f"({len(local_indices)} vs {len(remote_indices)})"
            )
        plan: list[tuple[int, int, int]] = []
        for li, ri in zip(local_indices, remote_indices, strict=True):
            local_addr, local_size, _ = local_dlist[li]
            remote_addr, remote_size, _ = remote_dlist[ri]
            assert local_size == remote_size, (
                f"FakeNixlWrapper: block-size mismatch "
                f"local={local_size} remote={remote_size}"
            )
            if op == "READ":
                # Pull remote (data owner) → local (initiator).
                plan.append((remote_addr, local_addr, local_size))
            else:  # WRITE: push local → remote.
                plan.append((local_addr, remote_addr, local_size))
        # The notification is delivered to the remote agent (the data owner)
        # when the transfer completes, regardless of direction.
        return _FakeXferHandle(plan, notif_msg, remote_agent)

    def transfer(self, handle: _FakeXferHandle) -> None:
        # Eager: complete the copy synchronously. The scheduler's poll loop
        # still runs unchanged; it just always sees DONE on the first tick.
        for src_addr, dst_addr, size in handle.plan:
            ctypes.memmove(dst_addr, src_addr, size)
        handle.state = "DONE"
        if handle.notif_msg:
            with _AGENTS_LOCK:
                owner = _AGENTS.get(handle.notify_agent)
                if owner is not None:
                    owner._notifs.append(handle.notif_msg)

    def get_new_notifs(self) -> dict[str, list[bytes]]:
        # Drain notifications addressed to this agent. The scheduler only
        # iterates the values, so the key is informational.
        with _AGENTS_LOCK:
            if not self._notifs:
                return {}
            drained = {self.name: list(self._notifs)}
            self._notifs.clear()
            return drained

    def check_xfer_state(self, handle: _FakeXferHandle) -> str:
        return handle.state

    def release_xfer_handle(self, handle: _FakeXferHandle) -> None:
        handle.released = True

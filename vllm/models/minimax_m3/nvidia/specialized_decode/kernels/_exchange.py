# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The boundary push exchange shared by producers and consumers.

Producers (o_proj GEMV, fused MoE engine) push their bf16 TP-partial rows into
every rank's exchange (the o_proj / MoE CuTe DSL kernels' push: one 16-byte chunk per
(row, 8 columns));
the next boundary consumer (the post kernel after o_proj, the next layer's
front after the MoE engine) polls its own copy, sums ranks in rank order and
re-arms the slots with the empty sentinel. One exchange per row bucket.

Layout (int32 words), identical on every rank::

    exch[2 sets][2 parities][maxt][world][H / 2]   bf16x2 words, kEmpty = 0x80000000
    flags[maxt][cluster]                           local only: per (row, column
                                                   slice) call parity, flipped by
                                                   the consumer

`PushParams` (160 bytes: peers[16], flags, rank, world, maxt, cluster, ndst)
is packed here; the post's host (`post/host.py`) unpacks the same bytes.

Multicast mode: when the transport gives the exchange an NVLS multicast
address, a producer stores each chunk once, to `mc_ptr` + slot (`peers[0]`
= the multicast address, `ndst` = 1), and the fabric delivers it to the same
slot of every rank; the unicast mode stores it `world` times. Consumers do
not change (they poll and re-arm their local copy). `multicast_policy`
picks the mode per exchange: multicast when it is available and the group
spans more than 4 ranks or the bucket is above 8 rows (unicast fan-out
costs world stores per chunk; NVLS one).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vllm.logger import init_logger

from ._comm import MAX_PEERS, PeerBuffer, PeerTransport

if TYPE_CHECKING:
    import torch

logger = init_logger(__name__)

HIDDEN = 6144
K_EMPTY = -0x80000000  # int32 0x80000000: two bf16 -0.0 halves
BUCKETS = (8, 16)  # row buckets: up to 8 rows, and up to 16
CLUSTER = 4  # exact-mode column slices per row
PUSH_PARAMS_FMT = (
    f"<{MAX_PEERS}qq5i"  # peers[16], flags, rank, world, maxt, cluster, ndst
)
PUSH_PARAMS_BYTES = 160  # the packed PushParams struct with 8-byte tail padding
MULTICAST_MIN_WORLD = 5  # multicast from TP8 up ...
MULTICAST_MIN_MAXT = 16  # ... or for buckets above 8 rows


def multicast_policy(world: int, maxt: int, available: bool) -> bool:
    """Whether an exchange of `world` ranks and bucket `maxt` pushes
    through its multicast address (the same answer on every rank)."""
    if not available or world == 1:
        return False
    return world >= MULTICAST_MIN_WORLD or maxt >= MULTICAST_MIN_MAXT


def bucket(rows: int) -> int:
    for b in BUCKETS:
        if rows <= b:
            return b
    raise ValueError(f"rows {rows} exceed the largest exchange bucket {BUCKETS[-1]}")


def exchange_words(world: int, maxt: int) -> int:
    return 4 * maxt * world * (HIDDEN // 2)


@dataclass
class PushWorkspace:
    maxt: int
    exch: PeerBuffer
    flags: torch.Tensor  # torch.int32 [maxt * CLUSTER], local
    multicast: bool = False  # producers push once, through exch.mc_ptr

    @property
    def rank(self) -> int:
        return self.exch.rank

    @property
    def world(self) -> int:
        return self.exch.world

    @property
    def ndst(self) -> int:
        """Destinations of one push: 1 (multicast) or world (unicast)."""
        return 1 if self.multicast else self.world

    def push_ptrs(self) -> list[int]:
        """The `ndst` addresses a producer stores each chunk to."""
        if self.multicast:
            if not self.exch.mc_ptr:
                raise ValueError(
                    f"{self.exch.name}: multicast without a multicast address"
                )
            return [int(self.exch.mc_ptr)]
        return [int(p) for p in self.exch.peer_ptrs]

    def params_bytes(self) -> bytes:
        dst = self.push_ptrs()
        b = struct.pack(
            PUSH_PARAMS_FMT,
            *(dst + [0] * (MAX_PEERS - len(dst))),
            int(self.flags.data_ptr()),
            self.rank,
            self.world,
            self.maxt,
            CLUSTER,
            self.ndst,
        )
        return b + b"\0" * (PUSH_PARAMS_BYTES - len(b))

    def params_tensor(self):
        import torch

        return torch.frombuffer(
            bytearray(self.params_bytes()), dtype=torch.uint8
        ).clone()

    def reset(self) -> None:
        """Re-arm (never during graph capture): every slot empty, parity 0."""
        self.exch.local.view(dtype=_i32()).fill_(K_EMPTY)
        self.flags.zero_()


def _i32():
    import torch

    return torch.int32


def push_workspace(
    transport: PeerTransport, maxt: int, tag: str = "g3", multicast: bool | None = None
) -> PushWorkspace:
    """The bucket-`maxt` exchange (allocated once per transport and tag).
    `multicast`: the push mode (None: `multicast_policy`); fixed by the
    first call for a (tag, maxt)."""
    import torch

    if maxt not in BUCKETS:
        raise ValueError(f"maxt {maxt} not in {BUCKETS}")
    if transport.world > MAX_PEERS:
        raise ValueError("world exceeds the device peer table")
    cache = transport.__dict__.setdefault("_push_ws", {})
    key = (tag, maxt)
    if key not in cache:
        exch = transport.alloc(
            f"{tag}.exch{maxt}", 4 * exchange_words(transport.world, maxt)
        )
        if multicast is None:
            multicast = multicast_policy(transport.world, maxt, bool(exch.mc_ptr))
        if multicast and not exch.mc_ptr:
            raise ValueError(
                f"{tag}: multicast push requested, the {transport.kind} transport has "
                "no multicast address"
            )
        ws = None
        # local; every rank then agrees (a collective), so none waits on a failed one
        try:
            flags = torch.zeros(
                maxt * CLUSTER, dtype=torch.int32, device=exch.local.device
            )
            ws = PushWorkspace(maxt, exch, flags, bool(multicast))
            ws.reset()
        except Exception:
            logger.exception(
                "%s exchange: setup failed on rank %d", tag, transport.rank
            )
        transport.agree(ws is not None, f"{tag} exchange setup")
        cache[key] = ws
    return cache[key]


__all__ = [
    "BUCKETS",
    "CLUSTER",
    "HIDDEN",
    "K_EMPTY",
    "PUSH_PARAMS_BYTES",
    "PushWorkspace",
    "bucket",
    "exchange_words",
    "multicast_policy",
    "push_workspace",
]

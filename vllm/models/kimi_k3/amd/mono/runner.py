# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The mono decode's process-wide state: one scratch buffer every launch
shares (their mailbox tags differ), the device step epoch and the TP group's
peer memory.

Mailbox tags carry the epoch the model bumps once a forward (``step_begin``,
which a CUDA graph captures), so no buffer is zeroed between steps.
"""

from typing import TYPE_CHECKING

import torch

from vllm.distributed import get_tp_group
from vllm.models.kimi_k3.amd.mono.common.tiles import MAX_TOKENS, TILE_TOKENS

if TYPE_CHECKING:
    from vllm.models.kimi_k3.amd.mono.common.peer_memory import PeerBuffer


class _State:
    scratch: torch.Tensor | None = None
    zero_bytes = 0
    epoch: torch.Tensor | None = None
    queue: torch.Tensor | None = None
    peers: "PeerBuffer | None" = None


def alloc_scratch(device) -> None:
    """At model init, outside any graph capture: the largest build's scratch
    and the epoch."""
    if _State.scratch is not None:
        return
    from vllm.models.kimi_k3.amd.mono import layer, overlap
    from vllm.models.kimi_k3.amd.mono.attention import kda
    from vllm.models.kimi_k3.amd.mono.stages import moe

    n = max(
        max(
            kda.scratch_bytes(
                kda.KdaPreBuild(
                    tokens=MAX_TOKENS // length * length,
                    qlen=length,
                    nblocks=8,
                    delta=True,
                    state_len=3 + TILE_TOKENS,
                )
            )
            for length in range(1, TILE_TOKENS + 1)
        ),
        moe.scratch_bytes(moe.MoeBuild(tokens=moe.MOE_TILE_TOKENS)),
        layer.scratch_bytes(layer.K2Build(tokens=moe.MOE_TILE_TOKENS, nblocks=8)),
        layer.scratch_bytes(
            layer.K2Build(
                tokens=48,
                nblocks=8,
                producer_latent_quant=True,
                stream_x=True,
                wide=True,
                tail_panels=True,
            )
        ),
        overlap.scratch_bytes(
            overlap.OverlapBuild(
                kda.KdaPreBuild(
                    tokens=16,
                    qlen=1,
                    nblocks=8,
                    delta=True,
                    state_len=3 + TILE_TOKENS,
                    state_rows=16,
                    pipeline=33562624,
                ),
                layer.K2Build(tokens=16, nblocks=8, producer_latent_quant=True),
                k1_workers=256,
                reuse_workers=True,
                tagged_core=True,
            )
        ),
    )
    _State.zero_bytes = (n + 255) // 256 * 256
    _State.scratch = torch.zeros(_State.zero_bytes, dtype=torch.uint8, device=device)
    _State.epoch = torch.zeros(1, dtype=torch.int32, device=device)
    _State.queue = torch.zeros(moe.QUEUE_BYTES // 4, dtype=torch.int32, device=device)


def step_begin() -> None:
    """Once a forward, before its first mono launch (a graph captures it)."""
    if _State.epoch is not None:
        _State.epoch.add_(1)


def peers(device) -> "PeerBuffer":
    """The TP group's peer buffer. Collective: every rank builds the same layers
    in the same order, so every rank reaches it at the first MoE layer."""
    if _State.peers is None:
        from vllm.models.kimi_k3.amd.mono import layer
        from vllm.models.kimi_k3.amd.mono.common.peer_memory import PeerBuffer
        from vllm.models.kimi_k3.amd.mono.stages import moe

        tp = get_tp_group()
        nbytes = max(moe.peer_bytes(moe.MOE_TILE_TOKENS), layer.peer_bytes(48))
        buf = PeerBuffer(nbytes, tp.cpu_group, tp.rank_in_group, tp.world_size, device)
        buf.bytes.zero_()
        _State.peers = buf
    return _State.peers


def launch_args() -> dict:
    """The state every launch takes."""
    assert _State.peers is not None and _State.epoch is not None
    return {
        "scratch": _State.scratch,
        "peers": _State.peers.addresses,
        "rank": get_tp_group().rank_in_group,
        "epoch": _State.epoch,
    }


def scratch_bytes() -> int:
    return _State.zero_bytes


def scratch() -> torch.Tensor:
    assert _State.scratch is not None
    return _State.scratch


def epoch() -> torch.Tensor:
    assert _State.epoch is not None
    return _State.epoch


def queue() -> torch.Tensor:
    assert _State.queue is not None
    return _State.queue

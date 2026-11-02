# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The specialized decode's per-rank state: one scratch buffer every launch shares
(their mailbox tags differ), the device step epoch, the health words and the TP
group's peer memory.

Mailbox tags carry the epoch the model bumps once a forward (``step_begin``,
which a CUDA graph captures), so no buffer is zeroed between steps.

The peer buffer's first ``PEER_HEAD`` bytes hold the output rank's flags, a
word a rank (``alarm``): a rank whose wait gives up sets its word there before it
stores anything else. The kernels see the buffer past the head.
"""

import torch

from vllm.distributed import get_dcp_group, get_tp_group
from vllm.models.kimi_k3.amd.specialized_decode.mono.abi import PEER_HEAD
from vllm.models.kimi_k3.amd.specialized_decode.mono.common.plan import HEALTH_WORDS
from vllm.models.kimi_k3.amd.specialized_decode.runtime import M1C_QLENS, MAX_TOKENS
from vllm.models.kimi_k3.amd.specialized_decode.runtime.peer import PeerBuffer


class MonoRunner:
    """Collective: every TP rank builds it at the same point."""

    def __init__(self, device: torch.device, dcp: int | None = None, m1: bool = True):
        """``dcp``: M1's ranks, 1 or the TP group (default: the DCP group's).
        ``m1``: the MLA layers run M1 (else vLLM's attention ops)."""
        from vllm.models.kimi_k3.amd.specialized_decode.mono import layer
        from vllm.models.kimi_k3.amd.specialized_decode.mono.attention import (
            kda,
            mla,
            mla_back,
        )

        self.scratch = torch.zeros(
            max(
                kda.scratch_bytes(
                    kda.KdaPreBuild(
                        tokens=MAX_TOKENS,
                        qlen=MAX_TOKENS,
                        nblocks=8,
                        delta=True,
                        state_len=3 + MAX_TOKENS,
                    )
                ),
                layer.scratch_bytes(layer.K2Build(tokens=MAX_TOKENS, nblocks=8)),
                mla.scratch_bytes(
                    mla.MlaPreBuild(tokens=MAX_TOKENS, nblocks=8, delta=True)
                ),
            ),
            dtype=torch.uint8,
            device=device,
        )
        self.epoch = torch.zeros(1, dtype=torch.int32, device=device)
        self.health = torch.zeros(HEALTH_WORDS, dtype=torch.int32, device=device)

        tp = get_tp_group()
        self.rank = tp.rank_in_group
        self.world = tp.world_size
        self.dcp = get_dcp_group().world_size if dcp is None else dcp
        # K2's regions, M1a's gathered query (QG), M1b's partials (CMB); M1
        # takes no peer memory at DCP1
        self.m1 = m1
        m1 = m1 and self.dcp > 1
        self.mla_peer_off = layer.peer_bytes(MAX_TOKENS)
        self.mla_post_off = self.mla_peer_off + (
            mla.peer_bytes(MAX_TOKENS, self.dcp) if m1 else 0
        )
        nbytes = (
            PEER_HEAD
            + self.mla_post_off
            + (mla_back.peer_bytes(MAX_TOKENS, self.dcp) if m1 else 0)
        )
        self.peers = PeerBuffer(nbytes, tp.cpu_group, self.rank, tp.world_size, device)
        self.peers.bytes.zero_()
        self.peer_table = self.peers.addresses + PEER_HEAD
        self.alarm = int(self.peers.addresses[0].item()) + 4 * self.rank
        self.flags = self.peers.bytes[: 4 * tp.world_size].view(torch.int32)
        self.fault: dict = {}  # K2Build's fault_* fields (tests only)
        self.fuse_m1b = True  # an MLA layer's M1b runs in its K2's launch
        self.first_layer = -1  # the step's first kernel layer: bumps the epoch
        self.kda_prefix = ""  # a KDA layer's name: the step's metadata
        self.mla_prefix = ""  # an M1 layer's name: its decode metadata
        self.interleave = 1  # the DCP KV cache interleave
        self.num_spec = 0  # speculative tokens a request

    @property
    def m1c(self) -> bool:
        """DCP1 at qlen 4 / 8 (M1b in K2): M1a writes a bf16 query, K2's launch
        runs the decode attention (M1c) first, its M1b merging the KV splits."""
        return self.dcp == 1 and self.num_spec + 1 in M1C_QLENS and self.fuse_m1b

    def step_begin(self) -> None:
        """Once a forward, before its first mono launch (a graph captures it)."""
        self.epoch.add_(1)

    def launch_args(self) -> dict:
        return {
            "scratch": self.scratch,
            "epoch": self.epoch,
            "health": self.health,
            "alarm": self.alarm,
        }

    def peer_args(self) -> dict:
        return {"peers": self.peer_table, "rank": self.rank}

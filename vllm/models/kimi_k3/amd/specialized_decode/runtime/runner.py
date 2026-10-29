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

from vllm.distributed import get_tp_group
from vllm.models.kimi_k3.amd.specialized_decode.mono.abi import PEER_HEAD
from vllm.models.kimi_k3.amd.specialized_decode.mono.common.plan import HEALTH_WORDS
from vllm.models.kimi_k3.amd.specialized_decode.runtime import MAX_QLEN, MAX_TOKENS
from vllm.models.kimi_k3.amd.specialized_decode.runtime.peer import PeerBuffer


class MonoRunner:
    """Collective: every TP rank builds it at the same point."""

    def __init__(self, device: torch.device):
        from vllm.models.kimi_k3.amd.specialized_decode.mono import layer
        from vllm.models.kimi_k3.amd.specialized_decode.mono.attention import kda

        # the widest K1 of every request length (its token tiles), the widest
        # K2 with every scratch region
        self.scratch = torch.zeros(
            max(
                *(
                    kda.scratch_bytes(
                        kda.KdaPreBuild(
                            tokens=MAX_TOKENS // q * q,
                            qlen=q,
                            nblocks=8,
                            delta=True,
                            state_len=3 + q,
                        )
                    )
                    for q in range(1, MAX_QLEN + 1)
                ),
                layer.scratch_bytes(
                    layer.K2Build(
                        tokens=MAX_TOKENS,
                        nblocks=8,
                        producer_latent_quant=True,
                        stream_x=True,
                    )
                ),
            ),
            dtype=torch.uint8,
            device=device,
        )
        self.epoch = torch.zeros(1, dtype=torch.int32, device=device)
        self.health = torch.zeros(HEALTH_WORDS, dtype=torch.int32, device=device)

        tp = get_tp_group()
        self.rank = tp.rank_in_group
        nbytes = PEER_HEAD + layer.peer_bytes(MAX_TOKENS)
        self.peers = PeerBuffer(nbytes, tp.cpu_group, self.rank, tp.world_size, device)
        self.peers.bytes.zero_()
        self.peer_table = self.peers.addresses + PEER_HEAD
        self.alarm = int(self.peers.addresses[0].item()) + 4 * self.rank
        self.flags = self.peers.bytes[: 4 * tp.world_size].view(torch.int32)
        self.fault: dict = {}  # K2Build's fault_* fields (tests only)
        self.first_layer = -1  # the step's first kernel layer: bumps the epoch
        self.kda_prefix = ""  # a KDA layer's name: the step's metadata
        self.num_spec = 0  # speculative tokens a request

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

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The specialized decode's per-rank state: one scratch buffer every launch shares
(their mailbox tags differ), the device step epoch, the health words and the TP
group's peer memory.

Mailbox tags carry the epoch the first kernel layer bumps once a step
(``step_begin``, which a CUDA graph captures), so no buffer is zeroed between
steps. The kernels mask the epoch to 22 bits, so the int32 counter may wrap.

The peer buffer's first ``PEER_HEAD`` bytes hold the output rank's flags, a word
a rank (``alarm``): a rank whose wait gives up sets its word there before it
stores anything else. The kernels see the buffer past the head.

Every size comes from ``mono.spec`` (no FlyDSL): the runner never imports the
kernel module.
"""

import torch

from vllm.distributed import get_tp_group
from vllm.logger import init_logger
from vllm.models.muse_spark.amd.specialized_decode.mono import spec
from vllm.models.muse_spark.amd.specialized_decode.mono.abi import (
    HEALTH_WORDS,
    PEER_HEAD,
    STARTUP_WAIT_SECONDS,
    WAIT_BUDGET,
)
from vllm.models.muse_spark.amd.specialized_decode.runtime.peer import PeerBuffer

logger = init_logger(__name__)

# host steps between two "served" log lines (one device read each)
LOG_EVERY = 10000
# bytes of the no-pad mask: >= MAX_TOKENS and whole words (ops.ld_pad reads the
# 32-bit word holding a row's byte)
NO_PAD_BYTES = 16


class MonoRunner:
    """Collective: every TP rank builds it at the same point."""

    def __init__(self, device: torch.device):
        self.scratch = torch.zeros(
            spec.scratch_bytes(), dtype=torch.uint8, device=device
        )
        self.epoch = torch.zeros(1, dtype=torch.int32, device=device)
        self.health = torch.zeros(HEALTH_WORDS, dtype=torch.int32, device=device)
        self.health[WAIT_BUDGET] = STARTUP_WAIT_SECONDS
        # Step.pad when the forward context carries no pad mask: no pad rows
        self.no_pad = torch.zeros(NO_PAD_BYTES, dtype=torch.uint8, device=device)

        tp = get_tp_group()
        self.rank = tp.rank_in_group
        nbytes = PEER_HEAD + spec.peer_bytes()
        self.peers = PeerBuffer(nbytes, tp.cpu_group, self.rank, tp.world_size, device)
        self.peers.bytes.zero_()
        self.peer_table = self.peers.addresses + PEER_HEAD
        self.alarm = int(self.peers.addresses[0].item()) + 4 * self.rank
        self.flags = self.peers.bytes[: 4 * tp.world_size].view(torch.int32)
        self.fault: dict = {}  # K2Build's fault_* fields (tests only)
        self.first_layer = -1  # the step's first kernel layer: bumps the epoch
        # the served-steps log (count_step)
        self.steps = 0
        self.epoch_mark: int | None = None

    def step_begin(self) -> None:
        """Once a forward, before its first mono launch (a graph captures it)."""
        self.epoch.add_(1)

    def set_wait_budget(self, seconds: int) -> None:
        """The seconds an in-kernel wait spins before it gives up (read once a
        launch): STARTUP_WAIT_SECONDS until the CUDA-graph capture ends, then
        WAIT_SECONDS."""
        self.health[WAIT_BUDGET : WAIT_BUDGET + 1].fill_(seconds)

    def reset_count(self) -> None:
        """Restarts the served-steps log at the device epoch (one device read):
        at the end of the startup (``prepare_kv``, ``after_capture``), so the
        self-test and warm-up steps are not counted as served."""
        self.steps = 0
        self.epoch_mark = int(self.epoch.item())

    def count_step(self) -> None:
        """Once per executed step (compute_logits). Every LOG_EVERY steps logs
        how many of them the kernels served: the device epoch's advance (one
        device read) over the host's step count, both since ``reset_count``."""
        if torch.cuda.is_current_stream_capturing():
            return
        if self.epoch_mark is None:
            self.epoch_mark = int(self.epoch.item())
        self.steps += 1
        if self.steps % LOG_EVERY:
            return
        epoch = int(self.epoch.item())
        served = (epoch - self.epoch_mark) % (1 << 32)  # the int32 counter wraps
        logger.info(
            "Muse Spark specialized decode: served %d of %d steps", served, self.steps
        )

    def launch_args(self) -> dict:
        return {
            "scratch": self.scratch,
            "epoch": self.epoch,
            "health": self.health,
            "alarm": self.alarm,
        }

    def peer_args(self) -> dict:
        return {"peers": self.peer_table, "rank": self.rank}

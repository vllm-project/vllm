# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runtime fault injection for `/ready` end-to-end tests.

Loaded into the server via ``--scheduler-cls`` / ``--worker-extension-cls``.
Scheduler faults are toggled with flag files under ``$VLLM_READY_FAULT_DIR``
(the scheduler lives in the EngineCore process). Worker faults are triggered
through the dev ``/collective_rpc`` endpoint, which only forwards strings.
"""

import os
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.v1.core.sched.async_scheduler import AsyncScheduler

FAULT_DIR_ENV = "VLLM_READY_FAULT_DIR"
SCHEDULER_STALL = "scheduler_stall"


def _launch_illegal_memory_access() -> None:
    import triton
    import triton.language as tl

    @triton.jit
    def _oob_store(ptr, offset):
        tl.store(ptr + offset, 1.0)

    buf = torch.empty(1, device="cuda", dtype=torch.float32)
    _oob_store[(1,)](buf, 1 << 42)


class FaultInjectionScheduler(AsyncScheduler):
    """While the ``scheduler_stall`` flag exists, schedule zero tokens.

    Mimics #45388: requests stay waiting but no tokens are ever scheduled,
    while the EngineCore busy loop keeps spinning.
    """

    max_num_scheduled_tokens: int

    def schedule(self, *args, **kwargs):
        fault_dir = os.environ.get(FAULT_DIR_ENV)
        if fault_dir is None or not os.path.exists(
            os.path.join(fault_dir, SCHEDULER_STALL)
        ):
            return super().schedule(*args, **kwargs)
        saved = self.max_num_scheduled_tokens
        self.max_num_scheduled_tokens = 0
        try:
            return super().schedule(*args, **kwargs)
        finally:
            self.max_num_scheduled_tokens = saved


class FaultInjectionWorkerExtension:
    vllm_config: VllmConfig
    model_runner: Any
    rank: int

    def _fault_targets_me(self, dp_ranks: str) -> bool:
        # Dense DP engines keep data_parallel_rank=0; the index is the DP rank.
        rank = self.vllm_config.parallel_config.data_parallel_index
        return rank in {int(r) for r in dp_ranks.split(",")}

    def fault_ima_now(self, dp_ranks: str) -> None:
        """Launch an illegal memory access without synchronizing.

        Models a latent async CUDA fault that has not surfaced yet because
        nothing synchronized afterwards.
        """
        if not self._fault_targets_me(dp_ranks):
            return
        _launch_illegal_memory_access()

    def fault_ima_now_on_worker(self, worker_rank: str) -> None:
        """Like ``fault_ima_now``, on one TP/PP worker of the engine."""
        if self.rank == int(worker_rank):
            _launch_illegal_memory_access()

    def fault_ima_on_next_forward(self, dp_ranks: str) -> None:
        """Launch an illegal memory access at the end of the next forward
        pass, so the fault belongs to the readiness probe's own GPU work."""
        if not self._fault_targets_me(dp_ranks):
            return

        def hook(*_):
            handle.remove()
            _launch_illegal_memory_access()

        handle = self.model_runner.get_model().register_forward_hook(hook)

    def fault_dp_mismatched_collective(self, dp_ranks: str) -> None:
        """Enter a DP all-reduce that the other ranks never join (#36594)."""
        if not self._fault_targets_me(dp_ranks):
            return
        from vllm.distributed.parallel_state import get_dp_group

        t = torch.ones(1, device="cuda")
        torch.distributed.all_reduce(t, group=get_dp_group().device_group)
        torch.accelerator.synchronize()

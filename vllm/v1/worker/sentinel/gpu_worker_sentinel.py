# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.config import ParallelConfig
from vllm.distributed import get_pp_group, get_tp_group
from vllm.v1.fault_tolerance import BaseSentinel


class WorkerSentinel(BaseSentinel):
    """
    Placeholder for WorkerSentinel.

    Current PR only implements fault-report plumbing; worker-side
    fault handling is intentionally left as a no-op. In the
    follow-up PR we will listen for commands from EngineCoreSentinel
    and execute reconfiguration/sleep/wake actions here.
    """

    def __init__(
        self,
        parallel_config: ParallelConfig,
        device: torch.device,
    ):
        dp_rank = parallel_config.data_parallel_rank
        tp_rank = get_tp_group().rank_in_group
        pp_rank = get_pp_group().rank_in_group
        identity_str = f"PP{pp_rank}_TP{tp_rank}"
        super().__init__(
            parallel_config, f"{dp_rank}_{identity_str}", identity_str.encode()
        )
        self.device = device
        torch.accelerator.set_device_index(self.device)

    def run(self) -> None:
        pass

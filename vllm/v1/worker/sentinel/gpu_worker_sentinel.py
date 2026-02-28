# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.config import VllmConfig
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
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        self.dp_rank = vllm_config.parallel_config.data_parallel_rank
        self.tp_rank = get_tp_group().rank_in_group
        self.pp_rank = get_pp_group().rank_in_group
        identity = f"PP{self.pp_rank}_TP{self.tp_rank}"
        super().__init__(
            sentinel_tag=f"{self.dp_rank}_{identity}",
            vllm_config=vllm_config,
        )
        self.device = device
        torch.cuda.set_device(self.device)

    def run(self) -> None:
        pass

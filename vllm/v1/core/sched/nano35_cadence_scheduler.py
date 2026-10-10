# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Experimental single-GPU prefill cadence using the existing throttle policy."""

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput


class Nano35CadenceScheduler(AsyncScheduler):
    prefill_interval = 4

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        return super().schedule(
            throttle_prefills=(
                throttle_prefills or self.current_step % self.prefill_interval != 0
            )
        )


class Nano35Cadence8Scheduler(Nano35CadenceScheduler):
    prefill_interval = 8

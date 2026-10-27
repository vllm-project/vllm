# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

from vllm.config.utils import config

if TYPE_CHECKING:
    from vllm.config.parallel import ParallelConfig


@config
class FaultToleranceConfig:
    """Configuration for fault tolerance."""

    engine_recovery_timeout_sec: int = 120
    """Timeout (in seconds) to wait for error handling instructions
    before raising an exception. If the EngineCore encounters an
    error, it waits up to this many seconds for vLLM to receive
    instructions on how to handle the error and then recover from the fault.
    If vLLM does not recover during this time, the original error is raised.
    """

    resume_requests_after_recovery: bool = False
    """If True, in-flight requests are preempted back to the waiting queue on
    fault (instead of being aborted) so they are automatically re-scheduled
    after recovery. Clients will not receive abort notifications.
    Setting this to False, or passing ``params={"abort": true}`` to the
    ``retry`` instruction, discards the in-flight requests instead.
    """


def ft_tp_barrier_required(parallel_config: "ParallelConfig") -> bool:
    """Whether the per-step FT barrier over the TP cpu group is armed.

    The barrier makes TP peers fail on the host within the gloo timeout
    instead of leaving an orphaned TP collective on the device stream.
    """
    return (
        parallel_config.enable_fault_tolerance
        and parallel_config.tensor_parallel_size > 1
    )

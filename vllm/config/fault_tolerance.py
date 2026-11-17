# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, Literal

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

    request_failure_policy: Literal["recompute", "fail"] = "fail"
    """Policy for handling unfinished requests when the engine encounters a fault.
    'recompute': preempt requests back to the waiting queue to recompute their
    KV cache and continue generation after recovery, without abort notifications.
    'fail': abort unfinished requests immediately (default).
    Passing `params={"abort": true}` to the `retry` instruction discards
    queued requests even when the policy is 'recompute'.
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

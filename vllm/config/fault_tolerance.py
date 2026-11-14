# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, Literal

from torch.distributed.constants import default_pg_timeout

import vllm.envs as envs
from vllm.config.utils import config
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config.parallel import ParallelConfig


logger = init_logger(__name__)

# Kernel timeouts of the FT all2all backends; vLLM cannot change them.
# DeepEP: NUM_TIMEOUT_CYCLES (200e9 cycles) at ~1.6 GHz. NIXL EP: Buffer
# timeout_ms.
_FT_KERNEL_TIMEOUT_SEC = {"deepep_low_latency": 130, "nixl_ep": 30}


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


def warn_ft_config_conflicts(pc: "ParallelConfig") -> None:
    """Warn when a timeout that surfaces a peer fault is too long for FT."""
    if not pc.enable_fault_tolerance:
        return

    in_step: dict[str, int] = {}
    if pc.all2all_backend in _FT_KERNEL_TIMEOUT_SEC:
        name = f"{pc.all2all_backend} kernel timeout"
        in_step[name] = _FT_KERNEL_TIMEOUT_SEC[pc.all2all_backend]
    if pc.world_size > 1 or pc.data_parallel_size > 1:
        cpu = pc.cpu_distributed_timeout_seconds
        if cpu is None:
            cpu = int(default_pg_timeout.total_seconds())
        in_step["cpu_distributed_timeout_seconds"] = cpu

    recovery = pc.fault_tolerance_config.engine_recovery_timeout_sec
    checks = [("engine_recovery_timeout_sec", recovery, in_step)]
    if pc.distributed_executor_backend == "mp":
        rpc = envs.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS
        reporting = {**in_step, "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS": rpc}
        checks = [
            ("engine_recovery_timeout_sec", recovery, reporting),
            ("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", rpc, in_step),
        ]
    for limit_name, limit, timeouts in checks:
        late = ", ".join(f"{n}={s}s" for n, s in timeouts.items() if s >= limit)
        if late:
            logger.warning(
                "Fault tolerance: %s not shorter than %s=%ds, so the fault "
                "surfaces too late. Lower them or raise %s.",
                late,
                limit_name,
                limit,
                limit_name,
            )

    communicator = pc.eplb_config.communicator
    if pc.enable_eplb and communicator in ("torch_nccl", "pynccl"):
        logger.warning(
            "Fault tolerance: EPLB communicator '%s' is NCCL-based, and the "
            "NCCL watchdog aborts the process when a collective fails. Use "
            "the 'nixl' or 'torch_gloo' communicator.",
            communicator,
        )

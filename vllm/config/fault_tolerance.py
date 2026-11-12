# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from typing import TYPE_CHECKING, Literal, NamedTuple

import vllm.envs as envs
from vllm.config.utils import config
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config.parallel import ParallelConfig


logger = init_logger(__name__)

# Timeouts in FT-adjacent components that vLLM does not expose.
_DEEPEP_LL_TIMEOUT_SEC = 100  # DeepEP NUM_TIMEOUT_CYCLES at about 2 GHz
_NIXL_EP_TIMEOUT_SEC = 30  # nixl_ep Buffer timeout_ms default (30000)
_GLOO_DEFAULT_TIMEOUT_SEC = 1800  # torch.distributed default_pg_timeout
_TCP_STORE_DEFAULT_TIMEOUT_SEC = 300  # torch.distributed.TCPStore default
_NIXL_EPLB_TIMEOUT_SEC = 300  # eplb_communicator._post_read_barrier

# NCCL aborts the process when a collective fails, which rules out recovery.
_NCCL_EPLB_COMMUNICATORS = frozenset({"torch_nccl", "pynccl"})


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


class FTTimeoutConflict(NamedTuple):
    """A timeout that can delay fault reporting past the FT recovery window."""

    name: str
    seconds: int
    source: str


class _FTTimeoutRule(NamedTuple):
    name: str
    applies: Callable[["ParallelConfig"], bool]
    seconds: Callable[["ParallelConfig"], int]
    source: Callable[["ParallelConfig"], str]
    under_execute_model: bool


def _configured_or_gloo_timeout_sec(parallel_config: "ParallelConfig") -> int:
    configured = parallel_config.cpu_distributed_timeout_seconds
    return configured if configured is not None else _GLOO_DEFAULT_TIMEOUT_SEC


# Fault tolerance accepts only the deepep_low_latency and nixl_ep all2all
# backends (gpu_worker_sentinel.FT_BACKEND_SET).
_FT_TIMEOUT_RULES: tuple[_FTTimeoutRule, ...] = (
    _FTTimeoutRule(
        name="DeepEP low-latency kernel timeout",
        applies=lambda pc: pc.all2all_backend == "deepep_low_latency",
        seconds=lambda pc: _DEEPEP_LL_TIMEOUT_SEC,
        source=lambda pc: "DeepEP NUM_TIMEOUT_CYCLES; not configurable from vLLM",
        under_execute_model=True,
    ),
    _FTTimeoutRule(
        name="NIXL EP kernel timeout",
        applies=lambda pc: pc.all2all_backend == "nixl_ep",
        seconds=lambda pc: _NIXL_EP_TIMEOUT_SEC,
        source=lambda pc: (
            "nixl_ep Buffer timeout_ms default; not configurable from vLLM"
        ),
        under_execute_model=True,
    ),
    _FTTimeoutRule(
        name="CPU distributed group timeout",
        applies=lambda pc: pc.world_size > 1 or pc.data_parallel_size > 1,
        seconds=_configured_or_gloo_timeout_sec,
        source=lambda pc: (
            "--cpu-distributed-timeout-seconds"
            if pc.cpu_distributed_timeout_seconds is not None
            else "PyTorch gloo default; --cpu-distributed-timeout-seconds unset"
        ),
        under_execute_model=True,
    ),
    _FTTimeoutRule(
        name="execute_model RPC timeout",
        applies=lambda pc: pc.tensor_parallel_size > 1,
        seconds=lambda pc: envs.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS,
        source=lambda pc: "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS",
        under_execute_model=False,
    ),
    _FTTimeoutRule(
        name="Worker recovery store timeout",
        applies=lambda pc: True,
        seconds=lambda pc: _TCP_STORE_DEFAULT_TIMEOUT_SEC,
        source=lambda pc: "PyTorch TCPStore default; not configurable from vLLM",
        under_execute_model=False,
    ),
    _FTTimeoutRule(
        name="Startup DP store timeout",
        applies=lambda pc: pc.data_parallel_size > 1,
        seconds=lambda pc: _GLOO_DEFAULT_TIMEOUT_SEC,
        source=lambda pc: (
            "PyTorch gloo default; group timeouts start after steady state"
        ),
        under_execute_model=False,
    ),
    _FTTimeoutRule(
        name="NIXL EPLB post-read barrier timeout",
        applies=lambda pc: pc.enable_eplb and pc.eplb_config.communicator == "nixl",
        seconds=lambda pc: _NIXL_EPLB_TIMEOUT_SEC,
        source=lambda pc: (
            "hard-coded in eplb_communicator._post_read_barrier; "
            "not configurable from vLLM"
        ),
        under_execute_model=False,
    ),
)


def ft_timeout_conflicts(
    parallel_config: "ParallelConfig",
) -> list[FTTimeoutConflict]:
    """Return FT-related timeouts at least as long as the recovery timeout.

    Args:
        parallel_config: Parallel configuration for the deployment.

    Returns:
        The offending timeouts in display order. Empty when fault tolerance is
        disabled or every applicable timeout is shorter than the recovery
        timeout.

    """
    if not parallel_config.enable_fault_tolerance:
        return []
    recovery = parallel_config.fault_tolerance_config.engine_recovery_timeout_sec
    conflicts: list[FTTimeoutConflict] = []
    for rule in _FT_TIMEOUT_RULES:
        if not rule.applies(parallel_config):
            continue
        seconds = rule.seconds(parallel_config)
        if seconds >= recovery:
            conflicts.append(
                FTTimeoutConflict(rule.name, seconds, rule.source(parallel_config))
            )
    return conflicts


def ft_execute_model_timeout_conflicts(
    parallel_config: "ParallelConfig",
) -> list[FTTimeoutConflict]:
    """Return timeouts that are not shorter than the execute_model RPC timeout.

    The RPC wraps a model step, so a collective or kernel that blocks longer
    than the RPC timeout fails the RPC before the fault surfaces.

    Args:
        parallel_config: Parallel configuration for the deployment.

    Returns:
        The offending timeouts in display order. Empty when fault tolerance or
        the RPC timeout does not apply.

    """
    if not (
        parallel_config.enable_fault_tolerance
        and parallel_config.tensor_parallel_size > 1
    ):
        return []
    rpc_timeout = envs.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS
    conflicts: list[FTTimeoutConflict] = []
    for rule in _FT_TIMEOUT_RULES:
        if not rule.under_execute_model or not rule.applies(parallel_config):
            continue
        seconds = rule.seconds(parallel_config)
        if seconds >= rpc_timeout:
            conflicts.append(
                FTTimeoutConflict(rule.name, seconds, rule.source(parallel_config))
            )
    return conflicts


def warn_ft_config_conflicts(parallel_config: "ParallelConfig") -> None:
    """Warn about fault-tolerance settings that prevent recovery.

    Args:
        parallel_config: Parallel configuration for the deployment.

    """
    if not parallel_config.enable_fault_tolerance:
        return

    recovery = parallel_config.fault_tolerance_config.engine_recovery_timeout_sec
    conflicts = ft_timeout_conflicts(parallel_config)
    if conflicts:
        details = "\n".join(
            f"  - {c.name}: {c.seconds} s ({c.source})" for c in conflicts
        )
        logger.warning(
            "Fault tolerance is enabled with engine_recovery_timeout_sec=%d. "
            "These timeouts are at least as long and can delay fault reporting "
            "past the recovery window:\n%s\n"
            "A fault surfaced only through one of these timeouts is observed "
            "after the recovery window has expired. Lower the configurable "
            "timeouts or raise engine_recovery_timeout_sec above them.",
            recovery,
            details,
        )

    order_conflicts = ft_execute_model_timeout_conflicts(parallel_config)
    if order_conflicts:
        details = "\n".join(
            f"  - {c.name}: {c.seconds} s ({c.source})" for c in order_conflicts
        )
        logger.warning(
            "Fault tolerance is enabled with VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS"
            "=%d. These timeouts are at least as long, so the execute_model "
            "RPC fails before the timeout that reports the peer fault "
            "expires:\n%s\n"
            "Raise VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS above them or lower "
            "the configurable timeouts.",
            envs.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS,
            details,
        )

    communicator = parallel_config.eplb_config.communicator
    if parallel_config.enable_eplb and communicator in _NCCL_EPLB_COMMUNICATORS:
        logger.warning(
            "Fault tolerance is enabled with EPLB communicator '%s'. The NCCL "
            "watchdog aborts the process on a collective failure, so fault "
            "tolerance cannot recover the rank. Use the 'nixl' or 'torch_gloo' "
            "communicator.",
            communicator,
        )

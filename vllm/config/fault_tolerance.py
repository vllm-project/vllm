# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from vllm.config.utils import config


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

    auto_recovery: bool = False
    """When enabled, the engine automatically recovers from faults
    without waiting for external orchestrator commands. On fault,
    it queries the worker's all2all mask: if all zeros, it retries; 
    if non-zero (dead peer detected), it performs scale-down.
    """

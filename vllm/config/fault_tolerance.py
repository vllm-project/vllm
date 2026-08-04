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

    enable_nan_fault_tolerance: bool = False
    """When enabled, requests producing NaN logits are immediately aborted and
    KV cache blocks are zeroed on reuse to prevent corruption from propagating
    into prefix cache, CPU/disk offloading tiers, or disaggregated decode
    instances. Implies --enable-detect-nans-in-logits."""

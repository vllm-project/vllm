# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from vllm.config.utils import config


@config
class FaultToleranceConfig:
    """Configuration for fault tolerance."""

    engine_recovery_timeout_sec: int = 60
    """Timeout (in seconds) to wait for error handling instructions
    before raising an exception. If the EngineCore encounters an
    error, it waits up to this many seconds for instructions on how
    to handle the error. If no instructions are received within this
    time, the original error is raised.
    """

    internal_fault_report_port: int = 22866
    """
    The port to use for engines to report fault to client sentinel.
    """

    external_fault_notify_port: int = 22867
    """
    Port used to publish engine fault and status change notifications.
    """

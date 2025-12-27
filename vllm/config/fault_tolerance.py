# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
from typing import Any

from pydantic.dataclasses import dataclass

from vllm.config.utils import config


@config
@dataclass
class FaultToleranceConfig:
    """Configuration for fault tolerance."""

    enable_fault_tolerance: bool = False
    """Enable fault tolerance for detailed error recovery,
    such as scaling down fault DPEngineCore.
    """

    shutdown_on_fault_tolerance_failure: bool = False
    """Whether to shut down vLLM when a fault tolerance action fails.
    """

    engine_recovery_timeout: int = 60
    """Timeout (in seconds) to wait for error handling instructions
    before raising an exception. If the EngineCore encounters an
    error, it waits up to this many seconds for instructions on how
    to handle the error. If no instructions are received within this
    time, the original error is raised.
    """

    internal_fault_report_port: int = 22866
    """
    The port to use for internal fault reporting.
    """

    external_fault_notify_port: int = 22867
    """
    The port to use for external fault notify.
    """

    gloo_comm_timeout: int = 30
    """
    The timeout for gloo communication.
    """

    worker_cmd_addr: str = None
    """
    ZMQ address used by EngineCoreSentinel to dispatch instructions to 
    WorkerSentinel instances. This address is assigned dynamically during 
    runtime.
    """

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.
        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        # no factors to consider.
        # this config will not affect the computation graph.
        factors: list[Any] = []
        hash_str = hashlib.md5(str(factors).encode(), usedforsecurity=False).hexdigest()
        return hash_str

    def __post_init__(self) -> None:
        pass

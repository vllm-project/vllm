# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
from typing import Any

from pydantic.dataclasses import dataclass

from vllm.config.utils import config


@config
@dataclass
class FaultToleranceConfig:
    """Configuration for distributed KV cache transfer."""

    enable_fault_tolerance: bool = False
    """Enable fault tolerance for detailed error recovery,
    such as scaling down fault DPEngineCore.
    """

    engine_recovery_timeout: int = 0
    """Timeout (in seconds) to wait for error handling instructions
    before raising an exception. If the EngineCore encounters an
    error, it waits up to this many seconds for instructions on how
    to handle the error. If no instructions are received within this
    time, the original error is raised.
    """

    fault_report_addr: str = "127.0.0.1"
    """
    The address to use for fault reporting.
    """

    fault_report_port: int = 22866
    """
    The port to use for fault reporting.
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

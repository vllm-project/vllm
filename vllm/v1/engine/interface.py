# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any, Protocol

from vllm.config import VllmConfig
from vllm.v1.executor.abstract import Executor


class IRunnableEngineCoreProc(Protocol):
    """
    Defines the basic contract for an engine that can be instantiated and run.
    The default, internal vLLM engines conform to this protocol.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_client: bool,
        handshake_address: str,
        executor_class: type[Executor],
        log_stats: bool,
        # ... other common args
    ) -> None:
        """Initializes the engine instance."""
        ...

    def run_busy_loop(self) -> None:
        """Runs the main loop of the engine, which blocks and processes
        requests."""
        ...

    @staticmethod
    def run_engine_core(*args: Any, **kwargs: Any) -> None:
        """The static entry point for launching the engine process."""
        ...


class IDiscoverableEngineCoreProc(IRunnableEngineCoreProc, Protocol):
    """
    Defines the contract for a pluggable engine that can be discovered by vLLM.
    External engines (e.g., from plugins) should conform to this protocol.
    """

    @staticmethod
    def is_supported() -> bool:
        """
        Returns True if this engine can run in the current environment.
        The implementation can check for hardware, environment variables, etc.
        """
        return False
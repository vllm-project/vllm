# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any, Protocol

from vllm.config import VllmConfig
from vllm.v1.executor.abstract import Executor


class IEngineCoreProc(Protocol):
    """An interface for a vLLM engine core process.

    This protocol defines the essential contract for any class that can be
    launched and managed as a vLLM engine. It ensures that the main vLLM
    process can interact with different engine implementations in a standardized
    way.
    """

    @staticmethod
    def is_supported() -> bool:
        """
        Returns True if this engine can run in the current environment.
        The implementation can check for hardware, environment variables, etc.
        """
        ...

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_client: bool,
        handshake_address: str,
        executor_class: type[Executor],
        log_stats: bool,
        # ... other common args
    ) -> None:
        """Initializes the engine instance.

        This is the fundamental constructor for any engine. The main vLLM
        process needs a standardized way to create an instance of the chosen
        engine, passing it the critical VllmConfig and other configuration
        objects. Without a consistent constructor signature, the registry
        pattern would be impossible.
        """
        ...

    def run_busy_loop(self) -> None:
        """Runs the main loop of the engine, which blocks and processes
        requests.

        This is the heart of the engine. It's the method that takes control of
        the process's lifecycle, continuously waits for work, processes
        requests, and drives the model. The launcher's job is to get the
        process to the point where it can call this method.
        """
        ...

    @staticmethod
    def run_engine_core(*args: Any, **kwargs: Any) -> None:
        """The static entry point for launching the engine process.

        This method is the single most important method for the DI pattern. It
        provides a static, universal entry point that the vLLM process launcher
        can call on the class itself before an instance even exists. It
        decouples the launcher from the specifics of how an engine is
        instantiated and run. The launcher's logic becomes:
        `EngineClass = registry.get("my_engine");
        EngineClass.run_engine_core(...)`
        """
        ...

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import abstractmethod

import zmq

from vllm.config import VllmConfig
from vllm.logger import init_logger

logger = init_logger(__name__)
# Polling timeout in milliseconds for non-blocking message reception
POLL_TIMEOUT_MS = 100


class BaseSentinel:
    """
    Abstract and constrain the core functionalities of the Sentinel.

    Core functionalities covered:
    - Fault listening
    - Fault tolerance instruction reception
    - Fault tolerance instruction execution
    - Upstream and downstream communication

    This class serves as the base abstraction for all LLM-related Sentinel
    implementations, enforcing standardized fault tolerance behavior across
    the system.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        sentinel_identity: bytes | None,
        sentinel_tag: str | None,
    ):
        self.sentinel_dead = False
        self.ctx = zmq.Context()
        self.sentinel_tag = sentinel_tag
        self.logger = self._make_logger()
        self.vllm_config = vllm_config
        self.ft_config = vllm_config.fault_tolerance_config

    def _make_logger(self):
        def log(msg, *args, level="info", **kwargs):
            """
            level: "info", "warning", "error", "debug"
            msg: log message
            """
            prefix = self.sentinel_name
            getattr(logger, level)(prefix + msg, *args, **kwargs)

        return log

    @property
    def sentinel_name(self) -> str:
        if self.sentinel_tag is None:
            return f"[{self.__class__.__name__}] "
        return f"[{self.__class__.__name__}_{self.sentinel_tag}] "

    @abstractmethod
    def run(self) -> None:
        """
        The run() method is launched as a separate thread when a Sentinel
        instance is created.

        This background thread typically runs persistently to ensure real-time
        detection of errors and timely reception of fault tolerance instructions
        from upstream sentinels.
        """
        raise NotImplementedError

    def shutdown(self):
        if self.ctx is not None:
            self.ctx.term()
        self.sentinel_dead = True

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import abstractmethod

import zmq

from vllm.config import VllmConfig
from vllm.logger import init_logger

logger = init_logger(__name__)


class BaseSentinel:
    """
    Core functionalities of the sentinel covered:
    - Fault listening
    - Fault tolerance instruction reception
    - Fault tolerance instruction execution
    - Upstream and downstream communication
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        sentinel_tag: str | None = None,
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
        self.sentinel_dead = True
        self.ctx.term()

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod

import zmq

from vllm.logger import init_logger

logger = init_logger(__name__)


class BaseSentinel(ABC):
    """
    Core functionalities of the sentinel covered:
    - Fault listening
    - Fault tolerance instruction reception
    - Fault tolerance instruction execution
    - Upstream and downstream communication
    """

    def __init__(
        self,
        sentinel_tag: str | None,
        identity: bytes,
    ):
        self.sentinel_dead = False
        self.ctx = zmq.Context()
        self.sentinel_tag = sentinel_tag
        self.logger = self._make_logger()
        self.identity = identity

    def _make_logger(self):
        def log(msg, *args, level="info", **kwargs):
            """msg: log message"""
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
        Run continuously to listen for control commands or error signals;
        on receipt, execute the command or report the result.
        """
        raise NotImplementedError

    def shutdown(self):
        self.sentinel_dead = True
        self.ctx.term()

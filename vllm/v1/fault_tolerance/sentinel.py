# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import weakref

import zmq

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
        sentinel_tag: str | None,
        identity: bytes,
        host,
    ):
        self.sentinel_dead = False
        if not hasattr(self, "ctx"):
            self.ctx = zmq.Context()
        self.sentinel_tag = sentinel_tag
        self.identity = identity
        self._host_ref = weakref.ref(host)

    @property
    def sentinel_name(self) -> str:
        if self.sentinel_tag is None:
            return f"[{self.__class__.__name__}] "
        return f"[{self.__class__.__name__}_{self.sentinel_tag}] "

    def run(self) -> None:
        """
        Run continuously to listen for control commands or error signals;
        on receipt, execute the command or report the result.
        """
        raise NotImplementedError

    @property
    def host(self):
        host = self._host_ref()
        if host is None:
            raise RuntimeError(
                f"{self.__class__.__name__}'s host has been garbage collected."
            )
        return host

    def shutdown(self):
        self.sentinel_dead = True
        self.ctx.term()

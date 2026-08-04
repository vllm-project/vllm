"""Base handler class for benchmark operations (Strategy Pattern)"""

# SPDX-License-Identifier: Apache-2.0

# Standard
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Local
    from ..benchmark import TestData, ZMQControllerBenchmark


class SocketType(Enum):
    """Socket type for benchmark operations"""

    PUSH = auto()  # PUSH socket for fire-and-forget messages
    DEALER = auto()  # DEALER socket for async request-reply messages
    HEARTBEAT = auto()  # Dedicated heartbeat DEALER socket


class OperationHandler(ABC):
    """Base class for benchmark operation handlers (Strategy Pattern)"""

    @property
    @abstractmethod
    def operation_name(self) -> str:
        """Return the operation name for registration"""
        pass

    @abstractmethod
    def create_message(
        self, benchmark: "ZMQControllerBenchmark", test_data: "TestData"
    ) -> Any:
        """Create a message for this operation"""
        pass

    @abstractmethod
    def get_message_count(self, benchmark: "ZMQControllerBenchmark") -> int:
        """Get the number of messages in a single operation"""
        pass

    @property
    def socket_type(self) -> SocketType:
        """Return the socket type for this operation (default: PUSH)"""
        return SocketType.PUSH

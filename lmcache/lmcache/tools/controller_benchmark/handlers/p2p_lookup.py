"""P2P lookup operation handler"""

# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import TYPE_CHECKING, Any
import random

# First Party
from lmcache.v1.cache_controller.message import BatchedP2PLookupMsg

if TYPE_CHECKING:
    # Local
    from ..benchmark import TestData, ZMQControllerBenchmark

# Local
from .base import OperationHandler, SocketType


class P2PLookupHandler(OperationHandler):
    """Handler for P2P lookup operations (uses DEALER-ROUTER socket)"""

    @property
    def operation_name(self) -> str:
        return "p2p_lookup"

    @property
    def socket_type(self) -> SocketType:
        return SocketType.DEALER

    def create_message(
        self, benchmark: "ZMQControllerBenchmark", test_data: "TestData"
    ) -> Any:
        instance = random.choice(test_data.instances)
        worker = random.choice(test_data.workers)
        hashes = [
            random.choice(test_data.keys) for _ in range(benchmark.config.num_hashes)
        ]

        return BatchedP2PLookupMsg(
            hashes=hashes,
            instance_id=instance,
            worker_id=worker,
        )

    def get_message_count(self, benchmark: "ZMQControllerBenchmark") -> int:
        return benchmark.config.num_hashes

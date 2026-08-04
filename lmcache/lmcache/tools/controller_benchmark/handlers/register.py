"""Register operation handler"""

# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import TYPE_CHECKING, Any
import random

# First Party
from lmcache.v1.cache_controller.message import RegisterMsg

if TYPE_CHECKING:
    # Local
    from ..benchmark import TestData, ZMQControllerBenchmark

# Local
from .base import OperationHandler, SocketType


class RegisterHandler(OperationHandler):
    """Handler for register operations (DEALER-ROUTER mode)"""

    @property
    def operation_name(self) -> str:
        return "register"

    @property
    def socket_type(self) -> SocketType:
        return SocketType.DEALER

    def create_message(
        self, benchmark: "ZMQControllerBenchmark", test_data: "TestData"
    ) -> Any:
        instance = random.choice(test_data.instances)
        worker = random.choice(test_data.workers)
        ip, port = self._generate_random_endpoint()
        peer_ip, peer_port = self._generate_random_endpoint()
        peer_init_url = "tcp://%s:%d" % (peer_ip, peer_port)

        return RegisterMsg(
            instance_id=instance,
            worker_id=worker,
            ip=ip,
            port=port,
            peer_init_url=peer_init_url,
        )

    def get_message_count(self, benchmark: "ZMQControllerBenchmark") -> int:
        return 1

    @staticmethod
    def _generate_random_endpoint():
        """Generate random IP and port for test messages"""
        ip = "192.168.%d.%d" % (random.randint(1, 255), random.randint(1, 255))
        port = random.randint(10000, 60000)
        return ip, port

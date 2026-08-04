"""Admit operation handler"""

# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import TYPE_CHECKING, Any
import random

# First Party
from lmcache.v1.cache_controller.message import (
    BatchedKVOperationMsg,
    KVOpEvent,
    OpType,
)

if TYPE_CHECKING:
    # Local
    from ..benchmark import TestData, ZMQControllerBenchmark

# Local
from .base import OperationHandler


class AdmitHandler(OperationHandler):
    """Handler for admit operations"""

    @property
    def operation_name(self) -> str:
        return "admit"

    def create_message(
        self, benchmark: "ZMQControllerBenchmark", test_data: "TestData"
    ) -> Any:
        instance = random.choice(test_data.instances)
        worker = random.choice(test_data.workers)
        location = random.choice(test_data.locations)

        operations = []
        for _ in range(benchmark.config.batch_size):
            key = random.choice(test_data.keys)
            seq_num = benchmark.get_next_sequence_number(instance, worker, location)
            operations.append(KVOpEvent(op_type=OpType.ADMIT, key=key, seq_num=seq_num))

        return BatchedKVOperationMsg(
            instance_id=instance,
            worker_id=worker,
            location=location,
            operations=operations,
        )

    def get_message_count(self, benchmark: "ZMQControllerBenchmark") -> int:
        return benchmark.config.batch_size

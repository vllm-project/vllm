# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput

pytestmark = pytest.mark.cpu_test


class DummyModelRunnerOutput(ModelRunnerOutput):
    def __init__(
        self,
        finished_sending: set[str] | None = None,
        finished_recving: set[str] | None = None,
        failed_recv_request_ids: set[str] | None = None,
        expected_finished_count: int = 0,
    ):
        self.kv_connector_output = KVConnectorOutput(
            finished_sending=finished_sending,
            finished_recving=finished_recving,
            failed_recv_request_ids=failed_recv_request_ids or set(),
            expected_finished_count=expected_finished_count,
        )

    def __repr__(self):
        return (
            f"DummyModelRunnerOutput("
            f"finished_sending={self.kv_connector_output.finished_sending},"
            f"finished_recving={self.kv_connector_output.finished_recving},"
            f"failed_recv_request_ids="
            f"{self.kv_connector_output.failed_recv_request_ids})"
        )


def test_aggregate_workers_output():
    aggregator = KVOutputAggregator(expected_finished_count=2)

    output1 = DummyModelRunnerOutput()
    output2 = DummyModelRunnerOutput()

    aggregated = aggregator.aggregate([output1, output2])

    assert aggregated is output1
    aggregated = aggregated.kv_connector_output
    assert aggregated.finished_sending is None
    assert aggregated.finished_recving is None
    assert not aggregated.failed_recv_request_ids

    output1 = DummyModelRunnerOutput(
        finished_sending={"req1"}, finished_recving={"req2"}
    )
    output2 = DummyModelRunnerOutput(failed_recv_request_ids={"req3"})

    aggregated = aggregator.aggregate([output1, output2])

    assert aggregated is output1
    aggregated = aggregated.kv_connector_output
    assert aggregated.finished_sending is None
    assert aggregated.finished_recving is None
    assert aggregated.failed_recv_request_ids == {"req3"}

    output1 = DummyModelRunnerOutput(failed_recv_request_ids={"req4"})
    output2 = DummyModelRunnerOutput(finished_sending={"req1"})

    aggregated = aggregator.aggregate([output1, output2])

    assert aggregated is output1
    aggregated = aggregated.kv_connector_output
    assert aggregated.finished_sending == {"req1"}
    assert aggregated.finished_recving is None
    assert aggregated.failed_recv_request_ids == {"req4"}

    output1 = DummyModelRunnerOutput(failed_recv_request_ids={"req5", "req6"})
    output2 = DummyModelRunnerOutput(
        finished_recving={"req2"}, failed_recv_request_ids={"req6", "req7"}
    )

    aggregated = aggregator.aggregate([output1, output2])

    assert aggregated is output1
    aggregated = aggregated.kv_connector_output
    assert aggregated.finished_sending is None
    assert aggregated.finished_recving == {"req2"}
    assert aggregated.failed_recv_request_ids == {"req5", "req6", "req7"}


def test_aggregate_workers_output_with_expected_finished_count():
    # We create the aggregator expecting to collect from 4 workers
    aggregator = KVOutputAggregator(expected_finished_count=4)
    assert aggregator._expected_finished_count == 4
    # Some request with default expected finished requests
    output1 = DummyModelRunnerOutput(finished_sending={"req1"})
    aggregated = aggregator.aggregate([output1])
    # still expecting to collect from 4 workers
    assert aggregator._send_remaining_count["req1"] == 3
    assert not aggregated.kv_connector_output.finished_sending
    assert not aggregated.kv_connector_output.finished_recving

    # Workers discover and find that in this setup they only need to
    # collect from 2
    output1 = DummyModelRunnerOutput(
        finished_sending={"req1"}, expected_finished_count=2
    )
    output2 = DummyModelRunnerOutput(
        finished_recving={"req2"}, expected_finished_count=2
    )
    output3 = DummyModelRunnerOutput(finished_recving={"req2"})
    # Req2 only needs 2 acks
    aggregated = aggregator.aggregate([output1, output2, output3])
    assert aggregated.kv_connector_output.expected_finished_count == 2

    assert not aggregated.kv_connector_output.finished_sending

    # Req2 is finished
    assert "req2" not in aggregator._recv_remaining_count
    assert aggregated.kv_connector_output.finished_recving == {"req2"}

    # Req1 is still waiting for 2 more acks (expected_finished_count has no effect)
    # NOTE: This is to showcase dynamic update. Workers are responsible for
    # ensuring "req1" termination in this case
    assert aggregator._send_remaining_count["req1"] == 2

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ECOutputAggregator."""

import pytest

from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorWorkerMetadata
from vllm.distributed.ec_transfer.ec_connector.utils import ECOutputAggregator
from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.v1.outputs import ECConnectorOutput, KVConnectorOutput, ModelRunnerOutput

pytestmark = pytest.mark.cpu_test


class FakeWorkerMeta(ECConnectorWorkerMetadata):
    """Records merge order. `aggregate` returns a new object, as the base class
    declares: an aggregator discarding the return value would lose the merge.
    """

    def __init__(self, saves: list[str]):
        self.saves = saves

    def aggregate(self, other: "FakeWorkerMeta") -> "FakeWorkerMeta":
        return FakeWorkerMeta(self.saves + other.saves)


def _worker_output(ec_output: ECConnectorOutput | None) -> ModelRunnerOutput:
    return ModelRunnerOutput(
        req_ids=[], req_id_to_index={}, ec_connector_output=ec_output
    )


def test_aggregate_folds_every_rank_onto_output_rank():
    """EC work done on any rank reaches the scheduler via output_rank's output.

    The middle rank reports no worker metadata: it must neither seed nor clobber
    the accumulator.
    """
    outputs = [
        _worker_output(
            ECConnectorOutput(
                finished_sending={"mm0"},
                ec_connector_worker_meta=FakeWorkerMeta(["mm0"]),
            )
        ),
        _worker_output(ECConnectorOutput(finished_recving={"mm1"})),
        _worker_output(
            ECConnectorOutput(ec_connector_worker_meta=FakeWorkerMeta(["mm2"]))
        ),
    ]

    result = ECOutputAggregator().aggregate(outputs, output_rank=2)

    assert result is outputs[2]
    assert result.ec_connector_output.finished_sending == {"mm0"}
    assert result.ec_connector_output.finished_recving == {"mm1"}
    assert result.ec_connector_output.ec_connector_worker_meta.saves == ["mm0", "mm2"]


def test_aggregate_leaves_no_ec_output_when_no_worker_reported():
    """Empty per-worker reports must not reach the scheduler as an empty object."""
    outputs = [_worker_output(ECConnectorOutput()), _worker_output(ECConnectorOutput())]

    result = ECOutputAggregator().aggregate(outputs, output_rank=0)

    assert result is outputs[0]
    assert result.ec_connector_output is None
    assert ECOutputAggregator().aggregate([None], output_rank=0) is None


def test_chaining_with_kv_aggregator_preserves_both_outputs():
    """MultiprocExecutor chains both aggregators and keeps only the last result,
    so each must merge onto the same output_rank output rather than replace it.
    """
    outputs = [
        _worker_output(ECConnectorOutput(finished_sending={"mm0"})),
        _worker_output(None),
    ]
    outputs[1].kv_connector_output = KVConnectorOutput(invalid_block_ids={7})

    result = None
    for aggregator in (
        KVOutputAggregator(expected_finished_count=1),
        ECOutputAggregator(),
    ):
        result = aggregator.aggregate(outputs, output_rank=1)

    assert result is outputs[1]
    assert result.kv_connector_output.invalid_block_ids == {7}
    assert result.ec_connector_output.finished_sending == {"mm0"}

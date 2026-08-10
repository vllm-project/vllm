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
    """Per-worker save/load reports, concatenated in merge order.

    `aggregate` returns a new object, as the base class declares: an
    aggregator that discarded the return value would lose the merge.
    """

    def __init__(
        self,
        completed_saves: list[str] | None = None,
        completed_loads: list[str] | None = None,
    ):
        self.completed_saves = completed_saves or []
        self.completed_loads = completed_loads or []

    def aggregate(self, other: "FakeWorkerMeta") -> "FakeWorkerMeta":
        return FakeWorkerMeta(
            self.completed_saves + other.completed_saves,
            self.completed_loads + other.completed_loads,
        )


def _worker_output(ec_output: ECConnectorOutput | None) -> ModelRunnerOutput:
    return ModelRunnerOutput(
        req_ids=[], req_id_to_index={}, ec_connector_output=ec_output
    )


def test_aggregate_unions_finished_ids_onto_output_rank():
    """EC work done on any rank reaches the scheduler via output_rank's output."""
    outputs = [
        _worker_output(
            ECConnectorOutput(finished_sending={"mm0"}, finished_recving={"mm1"})
        ),
        _worker_output(ECConnectorOutput(finished_sending={"mm2"})),
    ]

    result = ECOutputAggregator().aggregate(outputs, output_rank=1)

    assert result is outputs[1]
    assert result.ec_connector_output.finished_sending == {"mm0", "mm2"}
    assert result.ec_connector_output.finished_recving == {"mm1"}


def test_aggregate_worker_meta_folds_across_ranks():
    """Worker metadata is folded left across ranks, keeping each merge result."""
    outputs = [
        _worker_output(
            ECConnectorOutput(ec_connector_worker_meta=FakeWorkerMeta(["mm0"], []))
        ),
        _worker_output(
            ECConnectorOutput(ec_connector_worker_meta=FakeWorkerMeta([], ["mm1"]))
        ),
        _worker_output(
            ECConnectorOutput(ec_connector_worker_meta=FakeWorkerMeta(["mm2"], ["mm3"]))
        ),
    ]

    result = ECOutputAggregator().aggregate(outputs, output_rank=2)

    worker_meta = result.ec_connector_output.ec_connector_worker_meta
    assert worker_meta.completed_saves == ["mm0", "mm2"]
    assert worker_meta.completed_loads == ["mm1", "mm3"]


def test_aggregate_worker_meta_tolerates_ranks_without_meta():
    """Ranks reporting no metadata neither seed nor clobber the accumulator."""
    worker_meta = FakeWorkerMeta(["mm1"], [])
    outputs = [
        _worker_output(ECConnectorOutput(finished_sending={"mm0"})),
        _worker_output(ECConnectorOutput(ec_connector_worker_meta=worker_meta)),
        _worker_output(ECConnectorOutput(finished_recving={"mm2"})),
    ]

    result = ECOutputAggregator().aggregate(outputs, output_rank=0)

    assert result.ec_connector_output.ec_connector_worker_meta is worker_meta


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

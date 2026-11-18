# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The M-RoPE position delta a prefill instance reports to decode."""

from types import SimpleNamespace

import pytest

from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.utils import (
    MROPE_POSITION_DELTA_KEY,
    KVOutputAggregator,
    get_transferred_mrope_position_delta,
)
from vllm.sampling_params import SamplingParams
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput

from .utils import (
    create_model_runner_output,
    create_request,
    create_scheduler,
    create_vllm_config,
)

pytestmark = pytest.mark.cpu_test


def _vllm_config(kv_role: str) -> SimpleNamespace:
    return SimpleNamespace(
        kv_transfer_config=KVTransferConfig(
            kv_connector="NixlConnector", kv_role=kv_role
        )
    )


def _sampling_params(kv_transfer_params) -> SamplingParams:
    return SamplingParams(extra_args={"kv_transfer_params": kv_transfer_params})


@pytest.mark.parametrize(
    "reported, expected",
    [
        (-1980, -1980),
        (0, 0),
        # A protobuf Struct transport turns integers into floats.
        (-1980.0, -1980),
        (-1980.5, None),
        (True, None),
        ("-1980", None),
    ],
)
@pytest.mark.parametrize("kv_role", ["kv_consumer", "kv_both"])
def test_decode_reads_reported_delta(kv_role, reported, expected):
    params = _sampling_params(
        {"do_remote_prefill": True, MROPE_POSITION_DELTA_KEY: reported}
    )
    assert (
        get_transferred_mrope_position_delta(_vllm_config(kv_role), params) == expected
    )


def test_delta_ignored_without_kv_consumer_role():
    params = _sampling_params({MROPE_POSITION_DELTA_KEY: -1980})
    assert (
        get_transferred_mrope_position_delta(_vllm_config("kv_producer"), params)
        is None
    )
    assert (
        get_transferred_mrope_position_delta(
            SimpleNamespace(kv_transfer_config=None), params
        )
        is None
    )


@pytest.mark.parametrize(
    "sampling_params",
    [
        None,
        SamplingParams(),
        SamplingParams(extra_args={"other": 1}),
        _sampling_params(None),
        _sampling_params({"do_remote_prefill": True}),
    ],
)
def test_no_reported_delta(sampling_params):
    assert (
        get_transferred_mrope_position_delta(
            _vllm_config("kv_consumer"), sampling_params
        )
        is None
    )


@pytest.mark.parametrize("reported_delta", [-1980, None])
def test_prefill_reports_delta_in_kv_transfer_params(reported_delta):
    vllm_config = create_vllm_config()
    scheduler = create_scheduler(vllm_config)
    block_size = vllm_config.cache_config.block_size
    request = create_request(
        request_id=1,
        block_size=block_size,
        max_tokens=1,
        num_tokens=int(block_size * 2.5),
        do_remote_decode=True,
    )
    scheduler.add_request(request)

    scheduler_output = scheduler.schedule()
    model_runner_output = create_model_runner_output(reqs=[request])
    if reported_delta is not None:
        # The worker reports the delta in the same step that finishes the
        # request.
        model_runner_output.kv_connector_output = KVConnectorOutput(
            mrope_position_deltas={request.request_id: reported_delta}
        )
    engine_core_outputs = scheduler.update_from_output(
        scheduler_output, model_runner_output
    )

    assert request.is_finished()
    kv_transfer_params = engine_core_outputs[0].outputs[0].kv_transfer_params
    assert kv_transfer_params is not None
    assert kv_transfer_params["do_remote_prefill"]
    if reported_delta is None:
        assert MROPE_POSITION_DELTA_KEY not in kv_transfer_params
    else:
        assert kv_transfer_params[MROPE_POSITION_DELTA_KEY] == reported_delta


def test_reported_delta_ignored_for_unknown_request():
    vllm_config = create_vllm_config()
    scheduler = create_scheduler(vllm_config)
    request = create_request(request_id=1, max_tokens=1, do_remote_decode=True)
    scheduler.add_request(request)

    scheduler_output = scheduler.schedule()
    model_runner_output = create_model_runner_output(reqs=[request])
    model_runner_output.kv_connector_output = KVConnectorOutput(
        mrope_position_deltas={"unknown": -5}
    )
    engine_core_outputs = scheduler.update_from_output(
        scheduler_output, model_runner_output
    )

    kv_transfer_params = engine_core_outputs[0].outputs[0].kv_transfer_params
    assert kv_transfer_params is not None
    assert MROPE_POSITION_DELTA_KEY not in kv_transfer_params


def _worker_output(mrope_position_deltas) -> ModelRunnerOutput:
    return ModelRunnerOutput(
        req_ids=[],
        kv_connector_output=KVConnectorOutput(
            mrope_position_deltas=mrope_position_deltas
        ),
    )


def test_aggregator_merges_deltas_across_workers():
    aggregator = KVOutputAggregator(expected_finished_count=2)

    aggregated = aggregator.aggregate(
        [_worker_output({"req1": -3, "req2": 0}), _worker_output({"req1": -3})]
    )
    assert aggregated is not None
    assert aggregated.kv_connector_output is not None
    assert aggregated.kv_connector_output.mrope_position_deltas == {
        "req1": -3,
        "req2": 0,
    }

    aggregated = aggregator.aggregate([_worker_output(None), _worker_output(None)])
    assert aggregated is not None
    assert aggregated.kv_connector_output is not None
    assert aggregated.kv_connector_output.mrope_position_deltas is None


def test_deltas_make_kv_connector_output_non_empty():
    assert KVConnectorOutput().is_empty()
    assert not KVConnectorOutput(mrope_position_deltas={"req1": 0}).is_empty()

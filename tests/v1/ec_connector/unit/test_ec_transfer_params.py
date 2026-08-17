# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sanity tests for ec_transfer_params protocol plumbing.

No running engine required.
"""

from unittest.mock import MagicMock

from tests.v1.core.utils import create_scheduler
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.v1.request import Request, RequestStatus

EC_PARAMS: dict = {"mm_hash_abc": {"peer_host": "10.0.0.1", "peer_port": 5501}}


def test_ec_transfer_params_routed_to_sampling_params_extra_args():
    """ec_transfer_params on the request must land in SamplingParams.extra_args."""
    req = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=5,
        ec_transfer_params=EC_PARAMS,
    )
    sp = req.to_sampling_params(max_tokens=5, default_sampling_params={})
    assert sp.extra_args is not None
    assert sp.extra_args.get("ec_transfer_params") == EC_PARAMS


def test_request_output_add_propagates_ec_transfer_params():
    """RequestOutput.add() must carry ec_transfer_params forward to the caller."""

    def _out(ec_params):
        return RequestOutput(
            request_id="r1",
            prompt="p",
            prompt_token_ids=[1],
            prompt_logprobs=None,
            outputs=[
                CompletionOutput(
                    index=0,
                    text="",
                    token_ids=[],
                    cumulative_logprob=0.0,
                    logprobs=None,
                    finish_reason=None,
                )
            ],
            finished=False,
            ec_transfer_params=ec_params,
        )

    accumulated = _out(None)
    accumulated.add(_out(EC_PARAMS), aggregate=True)
    assert accumulated.ec_transfer_params == EC_PARAMS


def test_request_reads_ec_transfer_params_from_extra_args():
    """v1 Request must pull ec_transfer_params out of SamplingParams.extra_args."""
    sp = SamplingParams(extra_args={"ec_transfer_params": EC_PARAMS})
    req = Request(
        request_id="r1",
        prompt_token_ids=[1, 2, 3],
        sampling_params=sp,
        pooling_params=None,
    )
    assert req.ec_transfer_params == EC_PARAMS


def test_free_request_calls_ec_connector_and_surfaces_params():
    """_free_request must call ec_connector.request_finished() and return its params."""
    sp = SamplingParams(max_tokens=1)
    sp.update_from_generation_config({}, 50256)
    request = Request(
        request_id="test-req",
        prompt_token_ids=[1, 2, 3],
        sampling_params=sp,
        pooling_params=None,
        client_index=0,
    )
    scheduler = create_scheduler(use_ec_connector=True, ec_role="ec_producer")
    scheduler.add_request(request)
    request.status = RequestStatus.FINISHED_STOPPED

    mock_ec = MagicMock()
    mock_ec.request_finished.return_value = (False, EC_PARAMS)
    scheduler.ec_connector = mock_ec

    kv_params, ec_params = scheduler._free_request(request)

    mock_ec.request_finished.assert_called_once_with(request)
    assert ec_params == EC_PARAMS
    assert kv_params is None


def test_free_request_without_ec_connector_returns_none():
    """When no EC connector is configured, ec_transfer_params must be None."""
    sp = SamplingParams(max_tokens=1)
    sp.update_from_generation_config({}, 50256)
    request = Request(
        request_id="test-req",
        prompt_token_ids=[1, 2, 3],
        sampling_params=sp,
        pooling_params=None,
        client_index=0,
    )
    scheduler = create_scheduler(use_ec_connector=True, ec_role="ec_producer")
    scheduler.add_request(request)
    request.status = RequestStatus.FINISHED_STOPPED

    kv_params, ec_params = scheduler._free_request(request)

    assert ec_params is None
    assert kv_params is None

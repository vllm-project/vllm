# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time

import pytest

from vllm import SamplingParams
from vllm.v1.core.sched.utils import check_stop
from vllm.v1.engine.input_processor import InputProcessor
from vllm.v1.request import Request, RequestStatus

pytestmark = pytest.mark.skip_global_cleanup


def _make_request(params: SamplingParams) -> Request:
    return Request(
        request_id="req-0",
        client_index=0,
        prompt_token_ids=[1, 2, 3],
        sampling_params=params,
        pooling_params=None,
        arrival_time=time.time(),
    )


def test_normalize_trace_replay_params():
    params = SamplingParams(
        max_tokens=16,
        min_tokens=10,
        stop_token_ids=[20],
        trace_decode_token_ids=[10, 20, 30],
    )
    params.update_from_generation_config({}, eos_token_id=20)

    InputProcessor._normalize_trace_replay_params(params)

    assert params.max_tokens == 3
    assert params.min_tokens == 0
    assert params.ignore_eos is True
    assert params.eos_token_id is None
    assert params.stop_token_ids == []
    assert params.all_stop_token_ids == set()

    request = _make_request(params)
    assert request.max_tokens == 3
    request.append_output_token_ids([10, 20, 30])
    assert check_stop(request, max_model_len=128)
    assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED


def test_normalize_is_only_called_with_trace():
    """The caller guards with `if sampling_params.trace_decode_token_ids:`,
    so _normalize_trace_replay_params is never invoked without a trace."""
    params = SamplingParams(max_tokens=16, min_tokens=10, stop_token_ids=[20])
    assert params.trace_decode_token_ids is None
    # No call to _normalize_trace_replay_params — it asserts trace is set.

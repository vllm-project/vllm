# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from unittest.mock import MagicMock

import pytest

from vllm import SamplingParams
from vllm.v1.engine.core import EngineCore
from vllm.v1.request import Request

pytestmark = pytest.mark.skip_global_cleanup


def _make_engine_core(*, use_spec_decode: bool = False) -> EngineCore:
    engine_core = EngineCore.__new__(EngineCore)
    engine_core.use_spec_decode = use_spec_decode
    engine_core.scheduler = MagicMock()
    return engine_core


def _make_request(params: SamplingParams) -> Request:
    return Request(
        request_id="req-0",
        client_index=0,
        prompt_token_ids=[1, 2, 3],
        sampling_params=params,
        pooling_params=None,
        arrival_time=time.time(),
    )


def test_add_request_sets_trace_replay_generation_bounds():
    params = SamplingParams(
        max_tokens=16,
        ignore_eos=False,
        trace_decode_token_ids=[10, 20, 30],
    )
    request = _make_request(params)
    engine_core = _make_engine_core()

    engine_core.add_request(request)

    assert request.sampling_params is not params
    assert request.sampling_params.trace_decode_token_ids == [10, 20, 30]
    assert request.sampling_params.max_tokens == 3
    assert request.sampling_params.ignore_eos is True
    assert request.max_tokens == 3
    engine_core.scheduler.add_request.assert_called_once_with(request)


@pytest.mark.parametrize(
    ("use_spec_decode", "n"),
    [
        (True, 1),
        (False, 2),
    ],
)
def test_add_request_disables_trace_replay_for_incompatible_modes(
    use_spec_decode: bool,
    n: int,
):
    params = SamplingParams(
        max_tokens=16,
        n=n,
        ignore_eos=False,
        trace_decode_token_ids=[10, 20, 30],
    )
    request = _make_request(params)
    engine_core = _make_engine_core(use_spec_decode=use_spec_decode)

    engine_core.add_request(request)

    assert request.sampling_params is not params
    assert request.sampling_params.trace_decode_token_ids is None
    assert request.sampling_params.max_tokens == 16
    assert request.sampling_params.ignore_eos is False
    assert request.max_tokens == 16
    engine_core.scheduler.add_request.assert_called_once_with(request)

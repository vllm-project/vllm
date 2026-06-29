# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from array import array

import pytest

from vllm.sampling_params import SamplingParams
from vllm.v1.request import Request

pytestmark = pytest.mark.cpu_test


def test_request_token_histories_use_contiguous_arrays():
    request = Request(
        request_id="req-1",
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=8),
        pooling_params=None,
    )

    request.append_output_token_ids(4)
    request.append_output_token_ids([5, 6])

    assert isinstance(request._output_token_ids, array)
    assert isinstance(request._all_token_ids, array)
    assert list(request.output_token_ids) == [4, 5, 6]
    assert list(request.all_token_ids) == [1, 2, 3, 4, 5, 6]
    assert request.all_token_ids.copy() == [1, 2, 3, 4, 5, 6]

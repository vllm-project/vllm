# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import msgspec
import pytest

from vllm.outputs import RequestOutput
from vllm.v1.engine import EngineCoreOutput

pytestmark = pytest.mark.cpu_test


def test_request_output_forward_compatible():
    output = RequestOutput(
        request_id="test_request_id",
        prompt="test prompt",
        prompt_token_ids=[1, 2, 3],
        prompt_logprobs=None,
        outputs=[],
        finished=False,
        example_arg_added_in_new_version="some_value",
    )
    assert output is not None


def test_weight_version_output_propagation():
    core_output = EngineCoreOutput(
        request_id="request-1",
        new_token_ids=[1],
        weight_version="step-7",
    )
    decoded = msgspec.msgpack.decode(
        msgspec.msgpack.encode(core_output),
        type=EngineCoreOutput,
    )
    assert decoded.weight_version == "step-7"

    request_output = RequestOutput(
        request_id="request-1",
        prompt=None,
        prompt_token_ids=[1],
        prompt_logprobs=None,
        outputs=[],
        finished=True,
        weight_version=decoded.weight_version,
    )
    assert request_output.weight_version == "step-7"

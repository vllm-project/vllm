# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.tokenizers.deepseek_v4_encoding import (
    encode_arguments_to_dsml as encode_dsv4_arguments,
)
from vllm.tokenizers.deepseek_v32_encoding import (
    encode_arguments_to_dsml as encode_dsv32_arguments,
)


@pytest.mark.parametrize(
    "encoder",
    [encode_dsv4_arguments, encode_dsv32_arguments],
    ids=["deepseek_v4", "deepseek_v32"],
)
def test_deepseek_history_tool_arguments_must_be_object(encoder):
    with pytest.raises(ValueError, match="must be a JSON object"):
        encoder({"name": "noop", "arguments": "-12"})


@pytest.mark.parametrize(
    "encoder",
    [encode_dsv4_arguments, encode_dsv32_arguments],
    ids=["deepseek_v4", "deepseek_v32"],
)
def test_deepseek_history_tool_arguments_must_be_valid_json(encoder):
    with pytest.raises(ValueError, match="must be a valid JSON object"):
        encoder({"name": "noop", "arguments": "{-12}"})


@pytest.mark.parametrize(
    "encoder",
    [encode_dsv4_arguments, encode_dsv32_arguments],
    ids=["deepseek_v4", "deepseek_v32"],
)
def test_deepseek_history_tool_arguments_preserve_objects(encoder):
    rendered = encoder(
        {
            "name": "weather",
            "arguments": '{"city": "New York", "temp": 72, "alerts": ["wind"]}',
        }
    )

    assert '<｜DSML｜parameter name="city" string="true">New York' in rendered
    assert '<｜DSML｜parameter name="temp" string="false">72' in rendered
    assert '<｜DSML｜parameter name="alerts" string="false">["wind"]' in rendered

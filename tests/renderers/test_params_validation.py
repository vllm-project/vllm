# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.exceptions import VLLMValidationError
from vllm.renderers.params import TokenizeParams


def test_max_output_error_uses_public_name_once():
    """The message must name the limit with the public parameter name only.
    A `{var=}` self-documenting specifier used to leak the internal attribute
    name as well, producing 'max_model_len=max_total_tokens=8192'."""
    with pytest.raises(VLLMValidationError) as excinfo:
        TokenizeParams(
            max_total_tokens=8192,
            max_output_tokens=20000,
            max_output_tokens_param="max_tokens",
            max_total_tokens_param="max_model_len",
        )

    message = str(excinfo.value)
    assert "max_tokens=20000 cannot be greater than max_model_len=8192" in message
    assert "max_total_tokens" not in message


def test_max_output_error_with_default_names():
    with pytest.raises(VLLMValidationError) as excinfo:
        TokenizeParams(max_total_tokens=8192, max_output_tokens=20000)

    assert "cannot be greater than max_total_tokens=8192" in str(excinfo.value)

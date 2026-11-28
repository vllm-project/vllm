# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Ray Data LLM and similar tools import ``vllm.inputs.data.*``."""

import pytest

import vllm.inputs as vllm_inputs
from vllm.inputs import TextPrompt as TextPromptTop
from vllm.inputs import TokensPrompt as TokensPromptTop
from vllm.inputs.data import TextPrompt, TokensPrompt

pytestmark = pytest.mark.cpu_test


def test_inputs_data_submodule_exposed():
    assert hasattr(vllm_inputs, "data")
    assert TextPrompt is TextPromptTop
    assert TokensPrompt is TokensPromptTop

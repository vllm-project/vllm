# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm import SamplingParams

from .utils import greedy_sampling


@pytest.fixture
def sampling_config() -> SamplingParams:
    return greedy_sampling()


@pytest.fixture
def model_name() -> str:
    return "meta-llama/Llama-3.1-8B-Instruct"


@pytest.fixture(autouse=True)
def reset_torch_dynamo():
    yield
    torch._dynamo.reset()

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.sampling_params import SamplingParams


@pytest.mark.parametrize("model", ["distilbert/distilgpt2"])
@pytest.mark.parametrize("block_size", [16])
def test_computed_prefix_blocks(model: str, block_size: int):
    # This test checks if we are able to run the engine to completion
    # without triggering asserts.
    # We are in a scenario where all blocks from the second request's prompt
    # are full and already computed when the second request arrives.
    prompt = (
        "You are a helpful assistant. How do I build a car from cardboard and "
        "paper clips? Is there an easy to follow video tutorial available "
        "online for free?")
    prompt2 = (
        " Please recommend to me some resources where I can learn not only to "
        "handle technical difficulties of building a car, but also "
        "decoration.")

    engine_args = EngineArgs(model=model,
                             block_size=block_size,
                             enable_prefix_caching=True)

    engine = LLMEngine.from_engine_args(engine_args)
    sampling_params = SamplingParams()

    engine.add_request("0", prompt + prompt2, sampling_params)
    engine.step()
    engine.add_request("1", prompt, sampling_params)
    engine.step()

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm import SamplingParams

# Qwen3's tokenizer EOS is <|im_end|>; generation_config.json also lists
# <|endoftext|>, which vLLM merges into stop_token_ids.
MODEL = "Qwen/Qwen3-0.6B"
ENDOFTEXT = 151643


@pytest.mark.parametrize("multiprocessing", ["0", "1"])
def test_generation_config_eos_stops_generation(
    vllm_runner, monkeypatch, multiprocessing: str
):
    """An EOS id listed only in generation_config.json ends generation with
    both the in-process and the multiprocess engine core."""
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", multiprocessing)
    params = SamplingParams(temperature=0, max_tokens=8, logit_bias={ENDOFTEXT: 100.0})
    with vllm_runner(MODEL, enforce_eager=True) as runner:
        output = runner.llm.generate("Hello my name is", params)[0].outputs[0]

    assert list(output.token_ids) == [ENDOFTEXT]
    assert output.finish_reason == "stop"
    assert output.stop_reason == ENDOFTEXT

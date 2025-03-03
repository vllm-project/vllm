# SPDX-License-Identifier: Apache-2.0
import os

import pytest

from vllm import LLM, SamplingParams

if os.getenv("VLLM_USE_V1", "0") != "1":
    pytest.skip("Test package requires V1", allow_module_level=True)

MODEL = "meta-llama/Llama-3.2-1B"
PROMPT = "Hello my name is Robert and I"


@pytest.fixture(scope="module")
def model() -> LLM:
    return LLM(MODEL, enforce_eager=True)


def test_n_gt_1(model):
    """ParallelSampling is supported."""

    params = SamplingParams(n=3)
    outputs = model.generate(PROMPT, params)
    assert len(outputs[0].outputs) == 3


def test_best_of(model):
    """Raise a ValueError since best_of is deprecated."""

    params = SamplingParams(n=2, best_of=3)
    with pytest.raises(ValueError):
        _ = model.generate(PROMPT, params)


def test_penalties(model):
    """Check that we do not get errors if applied."""

    params = SamplingParams(
        temperature=1.2,
        presence_penalty=1.2,
        frequency_penalty=1.2,
        repetition_penalty=1.2,
        min_p=0.5,
        top_p=0.5,
        top_k=3,
    )
    _ = model.generate(PROMPT, params)


def test_stop(model):
    """Check that we respect the stop words."""

    output = model.generate(PROMPT, SamplingParams(temperature=0))
    split_text = output[0].outputs[0].text.split()

    STOP_IDX = 5
    params = SamplingParams(temperature=0, stop=split_text[STOP_IDX])
    output = model.generate(PROMPT, params)
    new_split_text = output[0].outputs[0].text.split()

    # Output should not contain the stop word.
    assert len(new_split_text) == STOP_IDX

    params = SamplingParams(temperature=0,
                            stop=split_text[STOP_IDX],
                            include_stop_str_in_output=True)
    output = model.generate(PROMPT, params)
    new_split_text = output[0].outputs[0].text.split()

    # Output should contain the stop word.
    assert len(new_split_text) == STOP_IDX + 1


def test_stop_token_ids(model):
    """Check that we respect the stop token ids."""

    output = model.generate(PROMPT, SamplingParams(temperature=0))

    stop_token_id_0 = output[0].outputs[0].token_ids[5]
    stop_token_id_1 = output[0].outputs[0].token_ids[6]

    stop_token_ids = [stop_token_id_1, stop_token_id_0]
    params = SamplingParams(temperature=0, stop_token_ids=stop_token_ids)
    output = model.generate(PROMPT, params)
    assert output[0].outputs[0].token_ids[-1] == stop_token_id_0

    stop_token_ids = [stop_token_id_0, stop_token_id_1]
    params = SamplingParams(temperature=0, stop_token_ids=stop_token_ids)
    assert output[0].outputs[0].token_ids[-1] == stop_token_id_0


def test_bad_words(model):
    """Check that we respect bad words."""

    with pytest.raises(ValueError):
        _ = model.generate(PROMPT, SamplingParams(bad_words=["Hello"]))

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm import LLM, SamplingParams

MODEL = "hmellor/tiny-random-LlamaForCausalLM"
PROMPT = "Hello my name is Robert and I"


@pytest.fixture(scope="module")
def llm() -> LLM:
    return LLM(MODEL, enforce_eager=True)


def test_n_gt_1(llm):
    """ParallelSampling is supported."""

    params = SamplingParams(n=3)
    outputs = llm.generate(PROMPT, params)
    assert len(outputs[0].outputs) == 3


def test_penalties(llm):
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
    _ = llm.generate(PROMPT, params)


def test_stop(llm):
    """Check that we respect the stop words."""

    output = llm.generate(PROMPT, SamplingParams(temperature=0))
    split_text = output[0].outputs[0].text.split()

    STOP_IDX = 5
    params = SamplingParams(temperature=0, stop=split_text[STOP_IDX])
    output = llm.generate(PROMPT, params)
    new_split_text = output[0].outputs[0].text.split()

    # Output should not contain the stop word.
    assert len(new_split_text) == STOP_IDX

    params = SamplingParams(
        temperature=0, stop=split_text[STOP_IDX], include_stop_str_in_output=True
    )
    output = llm.generate(PROMPT, params)
    new_split_text = output[0].outputs[0].text.split()

    # Output should contain the stop word.
    assert len(new_split_text) == STOP_IDX + 1


def test_stop_token_ids(llm):
    """Check that we respect the stop token ids."""

    output = llm.generate(PROMPT, SamplingParams(temperature=0))

    stop_token_id_0 = output[0].outputs[0].token_ids[5]
    stop_token_id_1 = output[0].outputs[0].token_ids[6]

    stop_token_ids = [stop_token_id_1, stop_token_id_0]
    params = SamplingParams(temperature=0, stop_token_ids=stop_token_ids)
    output = llm.generate(PROMPT, params)
    assert output[0].outputs[0].token_ids[-1] == stop_token_id_0

    stop_token_ids = [stop_token_id_0, stop_token_id_1]
    params = SamplingParams(temperature=0, stop_token_ids=stop_token_ids)
    output = llm.generate(PROMPT, params)
    assert output[0].outputs[0].token_ids[-1] == stop_token_id_0


def test_detokenize_false(llm):
    """Check that detokenize=False option works."""

    output = llm.generate(PROMPT, SamplingParams(detokenize=False))
    assert len(output[0].outputs[0].token_ids) > 0
    assert len(output[0].outputs[0].text) == 0

    output = llm.generate(
        PROMPT, SamplingParams(detokenize=False, logprobs=3, prompt_logprobs=3)
    )
    assert len(output[0].outputs[0].token_ids) > 0
    assert len(output[0].outputs[0].text) == 0

    prompt_logprobs = output[0].prompt_logprobs
    sampled_logprobs = output[0].outputs[0].logprobs
    assert len(prompt_logprobs) > 1
    assert len(sampled_logprobs) > 1
    for all_logprobs in (prompt_logprobs[1:], sampled_logprobs):
        for logprobs in all_logprobs:
            assert 3 <= len(logprobs) <= 4
            assert all(lp.decoded_token is None for lp in logprobs.values())


def test_bad_words(llm):
    """Check that we respect bad words."""

    output = llm.generate(PROMPT, SamplingParams(temperature=0))
    split_text = output[0].outputs[0].text.split()

    bad_words_1 = " ".join(split_text[:2])
    params = SamplingParams(temperature=0, bad_words=[bad_words_1])
    output = llm.generate(PROMPT, params)
    new_text = output[0].outputs[0].text
    assert bad_words_1 not in new_text

    bad_words_2 = new_text.split()[-1]
    params = SamplingParams(temperature=0, bad_words=[bad_words_1, bad_words_2])
    output = llm.generate(PROMPT, params)
    new_text = output[0].outputs[0].text
    assert bad_words_1 not in new_text
    assert bad_words_2 not in new_text


def test_logits_processor(llm):
    """Check that we reject logits processor."""

    # This sample logits processor gives infinite score to the i-th token,
    # where i is the length of the input sequence.
    # We therefore expect the output token sequence to be [0, 1, 2, ...]
    def pick_ith(token_ids, logits):
        logits[len(token_ids)] = float("inf")
        return logits

    with pytest.raises(ValueError):
        _ = llm.generate(PROMPT, SamplingParams(logits_processors=[pick_ith]))


def test_allowed_token_ids(llm):
    """Check that we can use allowed_token_ids."""

    TOKEN_ID = 10
    allowed_token_ids = [TOKEN_ID]
    output = llm.generate(PROMPT, SamplingParams(allowed_token_ids=allowed_token_ids))
    assert output[0].outputs[0].token_ids[-1] == TOKEN_ID

    # Reject empty allowed_token_ids.
    with pytest.raises(ValueError):
        _ = llm.generate(PROMPT, SamplingParams(allowed_token_ids=[]))

    # Reject negative token id.
    with pytest.raises(ValueError):
        _ = llm.generate(PROMPT, SamplingParams(allowed_token_ids=[-1]))

    # Reject out of vocabulary.
    with pytest.raises(ValueError):
        _ = llm.generate(PROMPT, SamplingParams(allowed_token_ids=[10000000]))


def test_seed(llm):
    """Check that seed impacts randomness."""

    out_1 = llm.generate(PROMPT, SamplingParams(seed=42))
    out_2 = llm.generate(PROMPT, SamplingParams(seed=42))
    out_3 = llm.generate(PROMPT, SamplingParams(seed=43))

    assert out_1[0].outputs[0].text == out_2[0].outputs[0].text
    assert out_1[0].outputs[0].text != out_3[0].outputs[0].text

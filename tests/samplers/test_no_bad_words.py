"""Make sure bad_words_ids works.

Run `pytest tests/samplers/test_no_bad_words.py`.
"""

import pytest
from typing import List, Tuple

from transformers import AutoTokenizer

from vllm import LLM, SamplingParams


MODELS = ["openai-community/gpt2"]


@pytest.mark.parametrize("model", MODELS)
def test_one_token_bad_word(
    model: str,
) -> None:
    llm = LLM(model=model)
    tokenizer = AutoTokenizer.from_pretrained(model, add_prefix_space=True)
    prompt = "Hi! How are"
    target_token = "you"
    target_token_id = tokenizer(target_token).input_ids[0]

    sampling_params = SamplingParams(temperature=0)
    output_text, output_token_ids = _generate(llm, prompt, sampling_params)

    assert output_text.startswith(f" {target_token}")
    assert target_token_id in output_token_ids

    sampling_params = SamplingParams(
        temperature=0,
        bad_words_ids=[[target_token_id]]
    )
    output_text, output_token_ids = _generate(llm, prompt, sampling_params)

    assert not output_text.startswith(f" {target_token}")
    assert target_token_id not in output_token_ids


@pytest.mark.parametrize("model", MODELS)
def test_two_token_bad_word(
    model: str,
) -> None:
    llm = LLM(model=model)
    tokenizer = AutoTokenizer.from_pretrained(model, add_prefix_space=True)
    prompt = "How old are you? I am 10"
    target_token1, target_token2, target_token3 = [
        "years", "old", "older"
    ]
    target_token_id1, target_token_id2, target_token_id3 = [
        tokenizer(target_token).input_ids[0]
        for target_token in [target_token1, target_token2, target_token3]
    ]

    sampling_params = SamplingParams(temperature=0)
    output_text, output_token_ids = _generate(llm, prompt, sampling_params)

    assert output_text.startswith(f" {target_token1} {target_token2}")
    assert target_token_id1 in output_token_ids
    assert target_token_id2 in output_token_ids

    sampling_params = SamplingParams(
        temperature=0,
        bad_words_ids=[[target_token_id1]]
    )
    output_text, output_token_ids = _generate(llm, prompt, sampling_params)

    assert not output_text.startswith(f" {target_token1}")
    assert target_token_id1 not in output_token_ids

    sampling_params = SamplingParams(
        temperature=0,
        bad_words_ids=[[target_token_id2]]
    )
    output_text, output_token_ids = _generate(llm, prompt, sampling_params)

    assert output_text.startswith(f" {target_token1}")
    assert not output_text.startswith(f" {target_token1} {target_token2}")
    assert target_token_id1 in output_token_ids
    assert target_token_id2 not in output_token_ids

    sampling_params = SamplingParams(
        temperature=0,
        bad_words_ids=[[target_token_id1, target_token_id2]]
    )
    output_text, output_token_ids = _generate(llm, prompt, sampling_params)

    assert output_text.startswith(f" {target_token1}")
    assert not output_text.startswith(f" {target_token1} {target_token2}")
    assert output_text.startswith(f" {target_token1} {target_token3}")
    assert target_token_id1 in output_token_ids
    assert target_token_id3 in output_token_ids

    sampling_params = SamplingParams(
        temperature=0,
        bad_words_ids=[[target_token_id1, target_token_id2], [target_token_id1, target_token_id3]]
    )
    output_text, output_token_ids = _generate(llm, prompt, sampling_params)

    assert output_text.startswith(f" {target_token1}")
    assert not output_text.startswith(f" {target_token1} {target_token2}")
    assert not output_text.startswith(f" {target_token1} {target_token3}")
    assert target_token_id1 in output_token_ids
    assert target_token_id3 in output_token_ids


def _generate(
    llm: LLM, prompt: str, sampling_params: SamplingParams
) -> Tuple[str, List[int]]:
    output = llm.generate(prompt, sampling_params=sampling_params)
    output_text = output[0].outputs[0].text
    output_token_ids = output[0].outputs[0].token_ids

    return output_text, output_token_ids

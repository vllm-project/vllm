"""Compare the with and without prefix caching.

Run `pytest tests/prefix_caching/test_multi_modal_prefix_caching.py`.
"""
from typing import Tuple

import pytest
from transformers import AutoTokenizer

from ..models.utils import check_logprobs_close

MODEL_NAME = "fixie-ai/ultravox-v0_3"
AUDIO_PLACEHOLDER = "<|reserved_special_token_0|>"


@pytest.fixture
def prompt():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    return tokenizer.apply_chat_template(
        [{
            "role": "system",
            "content": "You are a helpful assistant."
        }, {
            'role': 'user',
            'content': f"{AUDIO_PLACEHOLDER}\n\nDescribe the audio above."
        }],
        tokenize=False,
        add_generation_prompt=True)


@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("audio_asset_names",
                         [("winning_call", "mary_had_lamb")])
@pytest.mark.parametrize("num_logprobs", [5])
@pytest.mark.parametrize("max_tokens", [30])
def test_multi_modal_prefix_caching(
    vllm_runner,
    prompt: str,
    audio_asset_names: Tuple[str, str],
    dtype: str,
    num_logprobs: int,
    max_tokens: int,
) -> None:
    """
    Test the case when some sequences have the prefix cache hit
    and the others don't.
    """
    from vllm.assets.audio import AudioAsset

    audios = [
        AudioAsset(asset).audio_and_sample_rate for asset in audio_asset_names
    ]
    prompts = [prompt for _ in audios]

    with vllm_runner(
            MODEL_NAME,
            dtype=dtype,
            enable_prefix_caching=True,
    ) as vllm_model:
        # Run against the first prompt so the cache is populated
        _ = vllm_model.generate_greedy(prompts[:1],
                                       max_tokens,
                                       audios=audios[:1])

        # Run all the prompts
        with_prefix_caching = vllm_model.generate_greedy_logprobs(
            prompts, max_tokens, num_logprobs, audios=audios)

    with vllm_runner(
            MODEL_NAME,
            dtype=dtype,
            enable_prefix_caching=False,
    ) as vllm_model:
        # Run all the prompts
        without_prefix_caching = vllm_model.generate_greedy_logprobs(
            prompts, max_tokens, num_logprobs, audios=audios)

    check_logprobs_close(
        outputs_0_lst=with_prefix_caching,
        outputs_1_lst=without_prefix_caching,
        name_0="prefix_caching=True",
        name_1="prefix_caching=False",
    )

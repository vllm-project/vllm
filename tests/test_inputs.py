# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.config import ModelConfig
from vllm.inputs import zip_enc_dec_prompts
from vllm.inputs.preprocess import InputPreprocessor

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize(
    "mm_processor_kwargs,expected_mm_kwargs",
    [
        (None, [{}, {}]),
        ({}, [{}, {}]),
        ({"foo": 100}, [{"foo": 100}, {"foo": 100}]),
        ([{"foo": 100}, {"bar": 200}], [{"foo": 100}, {"bar": 200}]),
    ],
)
def test_zip_enc_dec_prompts(mm_processor_kwargs, expected_mm_kwargs):
    """Test mm_processor_kwargs init for zipping enc/dec prompts."""
    encoder_prompts = ["An encoder prompt", "Another encoder prompt"]
    decoder_prompts = ["A decoder prompt", "Another decoder prompt"]
    zipped_prompts = zip_enc_dec_prompts(
        encoder_prompts, decoder_prompts, mm_processor_kwargs
    )
    assert len(zipped_prompts) == len(encoder_prompts) == len(decoder_prompts)
    for enc, dec, exp_kwargs, zipped in zip(
        encoder_prompts, decoder_prompts, expected_mm_kwargs, zipped_prompts
    ):
        assert isinstance(zipped, dict)
        assert len(zipped.keys()) == 3
        assert zipped["encoder_prompt"] == enc
        assert zipped["decoder_prompt"] == dec
        assert zipped["mm_processor_kwargs"] == exp_kwargs


@pytest.mark.parametrize(
    "model_id",
    [
        "facebook/chameleon-7b",
    ],
)
@pytest.mark.parametrize(
    "prompt",
    [
        "",
        {"prompt_token_ids": []},
    ],
)
@pytest.mark.skip(
    reason=(
        "Applying huggingface processor on text inputs results in "
        "significant performance regression for multimodal models. "
        "See https://github.com/vllm-project/vllm/issues/26320"
    )
)
def test_preprocessor_always_mm_code_path(model_id, prompt):
    model_config = ModelConfig(model=model_id)
    input_preprocessor = InputPreprocessor(model_config)

    # HF processor adds sep token
    tokenizer = input_preprocessor.get_tokenizer()
    sep_token_id = tokenizer.vocab[tokenizer.sep_token]

    processed_inputs = input_preprocessor.preprocess(prompt)
    assert sep_token_id in processed_inputs["prompt_token_ids"]

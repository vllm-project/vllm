# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.config import ModelConfig
from vllm.inputs import zip_enc_dec_prompts
from vllm.inputs.parse import parse_raw_prompts
from vllm.inputs.preprocess import InputPreprocessor
from vllm.transformers_utils.tokenizer import init_tokenizer_from_configs

pytestmark = pytest.mark.cpu_test

STRING_INPUTS = [
    "",
    "foo",
    "foo bar",
    "foo baz bar",
    "foo bar qux baz",
]

TOKEN_INPUTS = [
    [-1],
    [1],
    [1, 2],
    [1, 3, 4],
    [1, 2, 4, 3],
]

INPUTS_SLICES = [
    slice(None, None, -1),
    slice(None, None, 2),
    slice(None, None, -2),
]


def test_parse_raw_single_batch_empty():
    with pytest.raises(ValueError, match="at least one prompt"):
        parse_raw_prompts([])

    with pytest.raises(ValueError, match="at least one prompt"):
        parse_raw_prompts([[]])


@pytest.mark.parametrize("string_input", STRING_INPUTS)
def test_parse_raw_single_batch_string_consistent(string_input: str):
    assert parse_raw_prompts(string_input) == parse_raw_prompts([string_input])


@pytest.mark.parametrize("token_input", TOKEN_INPUTS)
def test_parse_raw_single_batch_token_consistent(token_input: list[int]):
    assert parse_raw_prompts(token_input) == parse_raw_prompts([token_input])


@pytest.mark.parametrize("inputs_slice", INPUTS_SLICES)
def test_parse_raw_single_batch_string_slice(inputs_slice: slice):
    assert parse_raw_prompts(STRING_INPUTS)[inputs_slice] == parse_raw_prompts(
        STRING_INPUTS[inputs_slice]
    )


# yapf: disable
@pytest.mark.parametrize('mm_processor_kwargs,expected_mm_kwargs', [
    (None, [{}, {}]),
    ({}, [{}, {}]),
    ({"foo": 100}, [{"foo": 100}, {"foo": 100}]),
    ([{"foo": 100}, {"bar": 200}], [{"foo": 100}, {"bar": 200}]),
])
# yapf: enable
def test_zip_enc_dec_prompts(mm_processor_kwargs, expected_mm_kwargs):
    """Test mm_processor_kwargs init for zipping enc/dec prompts."""
    encoder_prompts = ['An encoder prompt', 'Another encoder prompt']
    decoder_prompts = ['A decoder prompt', 'Another decoder prompt']
    zipped_prompts = zip_enc_dec_prompts(encoder_prompts, decoder_prompts,
                                         mm_processor_kwargs)
    assert len(zipped_prompts) == len(encoder_prompts) == len(decoder_prompts)
    for enc, dec, exp_kwargs, zipped in zip(encoder_prompts, decoder_prompts,
                                            expected_mm_kwargs,
                                            zipped_prompts):
        assert isinstance(zipped, dict)
        assert len(zipped.keys()) == 3
        assert zipped['encoder_prompt'] == enc
        assert zipped['decoder_prompt'] == dec
        assert zipped['mm_processor_kwargs'] == exp_kwargs


@pytest.mark.parametrize("model_id", [
    "facebook/opt-125m",
])
@pytest.mark.parametrize("prompt", [
    {
        "prompt": "",
        "multi_modal_data": {
            "dummy": []
        },
    },
    {
        "prompt_token_ids": [],
        "multi_modal_data": {
            "dummy": []
        },
    },
])
def test_preprocessor_text_no_mm_inputs(model_id, prompt):
    model_config = ModelConfig(model=model_id)
    tokenizer = init_tokenizer_from_configs(model_config)
    input_preprocessor = InputPreprocessor(model_config, tokenizer)

    with pytest.raises(ValueError, match="does not support multimodal inputs"):
        input_preprocessor.preprocess(prompt)


@pytest.mark.parametrize("model_id", [
    "facebook/chameleon-7b",
])
@pytest.mark.parametrize("prompt", [
    "",
    {
        "prompt_token_ids": []
    },
])
def test_preprocessor_always_mm_code_path(model_id, prompt):
    model_config = ModelConfig(model=model_id)
    tokenizer = init_tokenizer_from_configs(model_config)
    input_preprocessor = InputPreprocessor(model_config, tokenizer)

    # HF processor adds sep token
    sep_token_id = tokenizer.vocab[tokenizer.sep_token]

    processed_inputs = input_preprocessor.preprocess(prompt)
    assert sep_token_id in processed_inputs["prompt_token_ids"]

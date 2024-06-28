"""
This test file includes some cases where it is inappropriate to
only get the `eos_token_id` from the tokenizer as defined by
:meth:`vllm.LLMEngine._get_eos_token_id`.
"""
from vllm.transformers_utils.config import try_get_generation_config
from vllm.transformers_utils.tokenizer import get_tokenizer


def test_get_llama3_eos_token():
    MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

    tokenizer = get_tokenizer(MODEL_NAME)
    assert tokenizer.eos_token_id == 128009

    generation_config = try_get_generation_config(MODEL_NAME)
    assert generation_config is not None
    assert generation_config.eos_token_id == [128001, 128009]


def test_get_blip2_eos_token():
    MODEL_NAME = "Salesforce/blip2-opt-2.7b"

    tokenizer = get_tokenizer(MODEL_NAME)
    assert tokenizer.eos_token_id == 2

    generation_config = try_get_generation_config(MODEL_NAME)
    assert generation_config is not None
    assert generation_config.eos_token_id == 50118

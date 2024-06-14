# This unit test should be moved to a new
# tests/test_guided_decoding directory.
import pytest
import torch
from transformers import AutoTokenizer

from vllm.model_executor.guided_decoding import (
    GuidedDecodingFields, get_guided_decoding_logits_processor)
from vllm.model_executor.guided_decoding.outlines_logits_processors import (
    JSONLogitsProcessor, RegexLogitsProcessor)


def test_guided_logits_processors(sample_regex, sample_json_schema):
    """Basic unit test for RegexLogitsProcessor and JSONLogitsProcessor."""
    tokenizer = AutoTokenizer.from_pretrained('HuggingFaceH4/zephyr-7b-beta')
    regex_LP = RegexLogitsProcessor(sample_regex, tokenizer)
    json_LP = JSONLogitsProcessor(sample_json_schema,
                                  tokenizer,
                                  whitespace_pattern=None)

    token_ids = tokenizer.encode(
        f"Give an example IPv4 address with this regex: {sample_regex}")
    tensor = torch.rand(32000)
    original_tensor = torch.clone(tensor)
    regex_LP(token_ids, tensor)
    assert tensor.shape == original_tensor.shape
    assert not torch.allclose(tensor, original_tensor)

    token_ids = tokenizer.encode(
        f"Give an employee profile that fits this schema: {sample_json_schema}"
    )
    tensor = torch.rand(32000)
    original_tensor = torch.clone(tensor)
    json_LP(token_ids, tensor)
    assert tensor.shape == original_tensor.shape
    assert not torch.allclose(tensor, original_tensor)


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["outlines", "lm-format-enforcer"])
async def test_guided_logits_processor_black_box(sample_regex,
                                                 sample_json_schema,
                                                 backend: str):
    tokenizer = AutoTokenizer.from_pretrained('HuggingFaceH4/zephyr-7b-beta')
    token_ids = tokenizer.encode(
        f"Give an example IPv4 address with this regex: {sample_regex}")
    regex_lp = get_guided_decoding_logits_processor(
        GuidedDecodingFields(guided_regex=sample_regex,
                             guided_decoding_backend=backend), tokenizer)
    assert regex_lp is not None
    tensor = torch.rand(32000)
    original_tensor = torch.clone(tensor)
    tensor = regex_lp(token_ids, tensor)
    assert tensor.shape == original_tensor.shape
    assert not torch.allclose(tensor, original_tensor)

    token_ids = tokenizer.encode(
        f"Give an employee profile that fits this schema: {sample_json_schema}"
    )
    json_lp = get_guided_decoding_logits_processor(
        GuidedDecodingFields(guided_json=sample_json_schema,
                             guided_decoding_backend=backend), tokenizer)
    assert json_lp is not None
    tensor = torch.rand(32000)
    original_tensor = torch.clone(tensor)
    tensor = json_lp(token_ids, tensor)
    assert tensor.shape == original_tensor.shape
    assert not torch.allclose(tensor, original_tensor)

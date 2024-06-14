from copy import deepcopy

import pytest
from transformers import AutoTokenizer

from tests.nm_utils.utils_skip import should_skip_test_group
from vllm.transformers_utils.tokenizer import get_cached_tokenizer

if should_skip_test_group(group_name="TEST_TOKENIZATION"):
    pytest.skip("TEST_TOKENIZATION=DISABLE, skipping tokenization test group",
                allow_module_level=True)


def test_cached_tokenizer():
    reference_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    reference_tokenizer.add_special_tokens({"cls_token": "<CLS>"})
    reference_tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<SEP>"]})
    cached_tokenizer = get_cached_tokenizer(deepcopy(reference_tokenizer))

    assert reference_tokenizer.encode("prompt") == cached_tokenizer.encode(
        "prompt")
    assert set(reference_tokenizer.all_special_ids) == set(
        cached_tokenizer.all_special_ids)
    assert set(reference_tokenizer.all_special_tokens) == set(
        cached_tokenizer.all_special_tokens)
    assert set(reference_tokenizer.all_special_tokens_extended) == set(
        cached_tokenizer.all_special_tokens_extended)

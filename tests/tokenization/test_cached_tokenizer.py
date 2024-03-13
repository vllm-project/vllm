from copy import deepcopy
from vllm.transformers_utils.tokenizer import _get_cached_tokenizer
from transformers import AutoTokenizer


def test_cached_tokenizer():
    reference_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    cached_tokenizer = _get_cached_tokenizer(deepcopy(reference_tokenizer))

    assert reference_tokenizer.encode("prompt") == cached_tokenizer.encode(
        "prompt")
    assert set(reference_tokenizer.all_special_ids) == set(
        cached_tokenizer.all_special_ids)
    assert set(reference_tokenizer.all_special_tokens) == set(
        cached_tokenizer.all_special_tokens)
    assert set(reference_tokenizer.all_special_tokens_extended) == set(
        cached_tokenizer.all_special_tokens_extended)

# SPDX-License-Identifier: Apache-2.0
import pickle
from copy import deepcopy

from transformers import AutoTokenizer

from vllm.transformers_utils.tokenizer import (AnyTokenizer,
                                               get_cached_tokenizer)


def test_cached_tokenizer():
    reference_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    reference_tokenizer.add_special_tokens({"cls_token": "<CLS>"})
    reference_tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<SEP>"]})

    cached_tokenizer = get_cached_tokenizer(deepcopy(reference_tokenizer))
    _check_consistency(cached_tokenizer, reference_tokenizer)

    pickled_tokenizer = pickle.dumps(cached_tokenizer)
    unpickled_tokenizer = pickle.loads(pickled_tokenizer)
    _check_consistency(unpickled_tokenizer, reference_tokenizer)


def _check_consistency(actual: AnyTokenizer, expected: AnyTokenizer):
    assert isinstance(actual, type(expected))

    assert actual.encode("prompt") == expected.encode("prompt")
    assert actual.all_special_ids == expected.all_special_ids
    assert actual.all_special_tokens == expected.all_special_tokens
    assert (actual.all_special_tokens_extended ==
            expected.all_special_tokens_extended)

# SPDX-License-Identifier: Apache-2.0

import pytest
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from vllm.lora.request import LoRARequest
from vllm.transformers_utils.tokenizer import get_lora_tokenizer
from vllm.transformers_utils.tokenizer_group import get_tokenizer_group

from ..conftest import get_tokenizer_pool_config


@pytest.mark.asyncio
@pytest.mark.parametrize("tokenizer_group_type", [None, "ray"])
async def test_tokenizer_group_lora(sql_lora_files, tokenizer_group_type):
    reference_tokenizer = AutoTokenizer.from_pretrained(sql_lora_files)
    tokenizer_group = get_tokenizer_group(
        get_tokenizer_pool_config(tokenizer_group_type),
        tokenizer_id="gpt2",
        enable_lora=True,
        max_num_seqs=1,
        max_loras=1,
        max_input_length=None,
    )
    lora_request = LoRARequest("1", 1, sql_lora_files)
    assert reference_tokenizer.encode("prompt") == tokenizer_group.encode(
        request_id="request_id", prompt="prompt", lora_request=lora_request)
    assert reference_tokenizer.encode(
        "prompt") == await tokenizer_group.encode_async(
            request_id="request_id",
            prompt="prompt",
            lora_request=lora_request)
    assert isinstance(tokenizer_group.get_lora_tokenizer(None),
                      PreTrainedTokenizerBase)
    assert tokenizer_group.get_lora_tokenizer(
        None) == await tokenizer_group.get_lora_tokenizer_async(None)

    assert isinstance(tokenizer_group.get_lora_tokenizer(lora_request),
                      PreTrainedTokenizerBase)
    assert tokenizer_group.get_lora_tokenizer(
        lora_request) != tokenizer_group.get_lora_tokenizer(None)
    assert tokenizer_group.get_lora_tokenizer(
        lora_request) == await tokenizer_group.get_lora_tokenizer_async(
            lora_request)


def test_get_lora_tokenizer(sql_lora_files, tmp_path):
    lora_request = None
    tokenizer = get_lora_tokenizer(lora_request)
    assert not tokenizer

    lora_request = LoRARequest("1", 1, sql_lora_files)
    tokenizer = get_lora_tokenizer(lora_request)
    assert tokenizer.get_added_vocab()

    lora_request = LoRARequest("1", 1, str(tmp_path))
    tokenizer = get_lora_tokenizer(lora_request)
    assert not tokenizer


@pytest.mark.parametrize("enable_lora", [True, False])
@pytest.mark.parametrize("max_num_seqs", [1, 2])
@pytest.mark.parametrize("max_loras", [1, 2])
def test_lora_tokenizers(enable_lora, max_num_seqs, max_loras):
    tokenizer_group = get_tokenizer_group(
        get_tokenizer_pool_config(None),
        tokenizer_id="gpt2",
        enable_lora=enable_lora,
        max_num_seqs=max_num_seqs,
        max_loras=max_loras,
        max_input_length=None,
    )
    if enable_lora:
        assert tokenizer_group.lora_tokenizers.capacity == max(
            max_num_seqs, max_loras)
    else:
        assert tokenizer_group.lora_tokenizers.capacity == 0

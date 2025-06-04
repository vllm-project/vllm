# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from vllm.transformers_utils.tokenizer_group import TokenizerGroup


@pytest.mark.asyncio
async def test_tokenizer_group():
    reference_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer_group = TokenizerGroup(
        tokenizer_id="gpt2",
        enable_lora=False,
        max_num_seqs=1,
        max_input_length=None,
    )
    assert reference_tokenizer.encode("prompt") == tokenizer_group.encode(
        prompt="prompt", lora_request=None)
    assert reference_tokenizer.encode(
        "prompt") == await tokenizer_group.encode_async(prompt="prompt",
                                                        lora_request=None)
    assert isinstance(tokenizer_group.get_lora_tokenizer(None),
                      PreTrainedTokenizerBase)
    assert tokenizer_group.get_lora_tokenizer(
        None) == await tokenizer_group.get_lora_tokenizer_async(None)

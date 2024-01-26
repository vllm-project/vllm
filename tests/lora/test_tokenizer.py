import pytest
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from vllm.lora.request import LoRARequest
from vllm.transformers_utils.tokenizer import TokenizerGroup, get_lora_tokenizer


@pytest.mark.asyncio
async def test_transformers_tokenizer():
    reference_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer = TokenizerGroup(
        tokenizer_id="gpt2",
        enable_lora=False,
        max_num_seqs=1,
        max_input_length=None,
    )
    assert reference_tokenizer.encode("prompt") == tokenizer.encode(
        request_id="request_id", prompt="prompt", lora_request=None)
    assert reference_tokenizer.encode(
        "prompt") == await tokenizer.encode_async(request_id="request_id",
                                                  prompt="prompt",
                                                  lora_request=None)
    assert isinstance(tokenizer.get_lora_tokenizer(None),
                      PreTrainedTokenizerBase)
    assert tokenizer.get_lora_tokenizer(
        None) == await tokenizer.get_lora_tokenizer_async(None)


@pytest.mark.asyncio
async def test_transformers_tokenizer_lora(sql_lora_files):
    reference_tokenizer = AutoTokenizer.from_pretrained(sql_lora_files)
    tokenizer = TokenizerGroup(
        tokenizer_id="gpt2",
        enable_lora=True,
        max_num_seqs=1,
        max_input_length=None,
    )
    lora_request = LoRARequest("1", 1, sql_lora_files)
    assert reference_tokenizer.encode("prompt") == tokenizer.encode(
        request_id="request_id", prompt="prompt", lora_request=lora_request)
    assert reference_tokenizer.encode(
        "prompt") == await tokenizer.encode_async(request_id="request_id",
                                                  prompt="prompt",
                                                  lora_request=lora_request)
    assert isinstance(tokenizer.get_lora_tokenizer(None),
                      PreTrainedTokenizerBase)
    assert tokenizer.get_lora_tokenizer(
        None) == await tokenizer.get_lora_tokenizer_async(None)

    assert isinstance(tokenizer.get_lora_tokenizer(lora_request),
                      PreTrainedTokenizerBase)
    assert tokenizer.get_lora_tokenizer(
        lora_request) != tokenizer.get_lora_tokenizer(None)
    assert tokenizer.get_lora_tokenizer(
        lora_request) == await tokenizer.get_lora_tokenizer_async(lora_request)


def test_get_lora_tokenizer(sql_lora_files, tmpdir):
    lora_request = None
    tokenizer = get_lora_tokenizer(lora_request)
    assert not tokenizer

    lora_request = LoRARequest("1", 1, sql_lora_files)
    tokenizer = get_lora_tokenizer(lora_request)
    assert tokenizer.get_added_vocab()

    lora_request = LoRARequest("1", 1, str(tmpdir))
    tokenizer = get_lora_tokenizer(lora_request)
    assert not tokenizer

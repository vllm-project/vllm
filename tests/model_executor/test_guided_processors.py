# SPDX-License-Identifier: Apache-2.0

import pickle

import pytest
import torch
from transformers import AutoTokenizer

from vllm.config import ModelConfig
from vllm.model_executor.guided_decoding import (
    get_guided_decoding_logits_processor,
    get_local_guided_decoding_logits_processor)
from vllm.model_executor.guided_decoding.outlines_logits_processors import (
    JSONLogitsProcessor, RegexLogitsProcessor)
from vllm.sampling_params import GuidedDecodingParams

MODEL_NAME = 'HuggingFaceH4/zephyr-7b-beta'
GUIDED_DECODING_BACKENDS = ["outlines", "lm-format-enforcer", "xgrammar"]


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
@pytest.mark.parametrize("backend", GUIDED_DECODING_BACKENDS)
@pytest.mark.parametrize("is_local", [True, False])
async def test_guided_logits_processor_black_box(backend: str, is_local: bool,
                                                 sample_regex,
                                                 sample_json_schema):

    config = ModelConfig(
        MODEL_NAME,
        task="generate",
        tokenizer=MODEL_NAME,
        tokenizer_mode="auto",
        trust_remote_code=False,
        seed=0,
        dtype="bfloat16",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    token_ids = tokenizer.encode(
        f"Give an example IPv4 address with this regex: {sample_regex}")
    regex_request = GuidedDecodingParams(regex=sample_regex, backend=backend)

    regex_lp = get_local_guided_decoding_logits_processor(
            regex_request, tokenizer, config) if is_local else \
            await get_guided_decoding_logits_processor(
                    regex_request, tokenizer, config)
    assert regex_lp is not None
    tensor = torch.rand(32000)
    original_tensor = torch.clone(tensor)
    tensor = regex_lp(token_ids, tensor)
    assert tensor.shape == original_tensor.shape
    assert not torch.allclose(tensor, original_tensor)

    token_ids = tokenizer.encode(
        f"Give an employee profile that fits this schema: {sample_json_schema}"
    )
    json_request = GuidedDecodingParams(json=sample_json_schema,
                                        backend=backend)
    json_lp = await get_guided_decoding_logits_processor(
        json_request, tokenizer, config)
    assert json_lp is not None
    tensor = torch.rand(32000)
    original_tensor = torch.clone(tensor)
    tensor = json_lp(token_ids, tensor)
    assert tensor.shape == original_tensor.shape
    assert not torch.allclose(tensor, original_tensor)


def test_multiple_guided_options_not_allowed(sample_json_schema, sample_regex):
    with pytest.raises(ValueError,
                       match="You can only use one kind of guided"):
        GuidedDecodingParams(json=sample_json_schema, regex=sample_regex)

    with pytest.raises(ValueError,
                       match="You can only use one kind of guided"):
        GuidedDecodingParams(json=sample_json_schema, json_object=True)

    with pytest.raises(ValueError,
                       match="You can only use one kind of guided"):
        GuidedDecodingParams(json=sample_json_schema, choice=["a", "b"])

    with pytest.raises(ValueError,
                       match="You can only use one kind of guided"):
        GuidedDecodingParams(json=sample_json_schema, grammar="test grammar")


def test_guided_decoding_backend_options():
    """Test backend-specific options"""
    params = GuidedDecodingParams(
        backend="xgrammar:option-1,option-2,option-3")
    assert params.backend_options() == ["option-1", "option-2", "option-3"]

    no_fallback = GuidedDecodingParams(backend="xgrammar:option-1,no-fallback")
    assert no_fallback.no_fallback()


def test_pickle_xgrammar_tokenizer_data():

    # TODO: move to another test file for xgrammar
    try:
        import xgrammar as xgr
    except ImportError:
        pytest.skip("Could not import xgrammar to run test")

    from vllm.model_executor.guided_decoding.xgrammar_decoding import (
        TokenizerData)
    tokenizer_data = TokenizerData(vocab_type=xgr.VocabType.RAW)
    pickled = pickle.dumps(tokenizer_data)

    assert pickled is not None

    depickled: TokenizerData = pickle.loads(pickled)

    assert depickled is not None
    assert depickled.vocab_type == xgr.VocabType.RAW

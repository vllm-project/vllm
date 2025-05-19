# SPDX-License-Identifier: Apache-2.0

import json
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
GUIDED_DECODING_BACKENDS = [
    "outlines", "lm-format-enforcer", "xgrammar", "guidance"
]
GUIDED_DECODING_BACKENDS_WITH_REASONING_SUPPORT = ["outlines", "xgrammar"]
REASONING_MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"


# Initialize the tokenizer for the model here to avoid repeated loading
@pytest.fixture(scope="module")
def zephyr_7B_tokenzer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


@pytest.fixture(scope="module")
def deepseek_r1_qwen_tokenizer():
    return AutoTokenizer.from_pretrained(REASONING_MODEL_NAME)


def test_guided_logits_processors(zephyr_7B_tokenzer, sample_regex,
                                  sample_json_schema):
    """Basic unit test for RegexLogitsProcessor and JSONLogitsProcessor."""
    regex_LP = RegexLogitsProcessor(sample_regex,
                                    zephyr_7B_tokenzer,
                                    reasoner=None)
    json_LP = JSONLogitsProcessor(sample_json_schema,
                                  zephyr_7B_tokenzer,
                                  whitespace_pattern=None,
                                  reasoner=None)

    token_ids = zephyr_7B_tokenzer.encode(
        f"Give an example IPv4 address with this regex: {sample_regex}")
    tensor = torch.rand(32000)
    original_tensor = torch.clone(tensor)
    regex_LP(token_ids, tensor)
    assert tensor.shape == original_tensor.shape
    assert not torch.allclose(tensor, original_tensor)

    token_ids = zephyr_7B_tokenzer.encode(
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
                                                 sample_json_schema,
                                                 zephyr_7B_tokenzer):

    config = ModelConfig(
        MODEL_NAME,
        task="generate",
        tokenizer=MODEL_NAME,
        tokenizer_mode="auto",
        trust_remote_code=False,
        seed=0,
        dtype="bfloat16",
    )
    token_ids = zephyr_7B_tokenzer.encode(
        f"Give an example IPv4 address with this regex: {sample_regex}")
    regex_request = GuidedDecodingParams(regex=sample_regex, backend=backend)

    regex_lp = get_local_guided_decoding_logits_processor(
            regex_request, zephyr_7B_tokenzer, config) if is_local else \
            await get_guided_decoding_logits_processor(
                    regex_request, zephyr_7B_tokenzer, config)
    assert regex_lp is not None
    tensor = torch.rand(32000)
    original_tensor = torch.clone(tensor)
    tensor = regex_lp(token_ids, tensor)
    assert tensor.shape == original_tensor.shape
    assert not torch.allclose(tensor, original_tensor)

    token_ids = zephyr_7B_tokenzer.encode(
        f"Give an employee profile that fits this schema: {sample_json_schema}"
    )
    json_request = GuidedDecodingParams(json=sample_json_schema,
                                        backend=backend)
    json_lp = await get_guided_decoding_logits_processor(
        json_request, zephyr_7B_tokenzer, config)
    assert json_lp is not None
    tensor = torch.rand(32000)
    original_tensor = torch.clone(tensor)
    tensor = json_lp(token_ids, tensor)
    assert tensor.shape == original_tensor.shape
    assert not torch.allclose(tensor, original_tensor)


@pytest.mark.asyncio
@pytest.mark.parametrize("backend",
                         GUIDED_DECODING_BACKENDS_WITH_REASONING_SUPPORT)
@pytest.mark.parametrize("is_local", [True, False])
@pytest.mark.parametrize("reasoning_backend", ["deepseek_r1"])
async def test_guided_logits_processor_with_reasoning(
        backend: str, is_local: bool, reasoning_backend: str, sample_regex,
        sample_json_schema, deepseek_r1_qwen_tokenizer):

    config = ModelConfig(
        REASONING_MODEL_NAME,
        task="generate",
        tokenizer=REASONING_MODEL_NAME,
        tokenizer_mode="auto",
        trust_remote_code=False,
        seed=0,
        dtype="bfloat16",
    )
    token_ids = deepseek_r1_qwen_tokenizer.encode(
        f"Give an example IPv4 address with this regex: {sample_regex}."
        "<think>here is the thinking process")
    regex_request = GuidedDecodingParams(regex=sample_regex, backend=backend)

    regex_lp = get_local_guided_decoding_logits_processor(regex_request,
                    deepseek_r1_qwen_tokenizer, config,
                    reasoning_backend) if is_local else \
            await get_guided_decoding_logits_processor(
                    regex_request, deepseek_r1_qwen_tokenizer, config,
                    reasoning_backend)
    assert regex_lp is not None
    tensor = torch.rand(32000)
    original_tensor = torch.clone(tensor)
    tensor = regex_lp(token_ids, tensor)
    assert tensor.shape == original_tensor.shape
    assert torch.allclose(tensor, original_tensor)

    token_ids = deepseek_r1_qwen_tokenizer.encode(
        f"Give an employee profile that fits this schema: {sample_json_schema}."
        "<think>here is the thinking process")
    json_request = GuidedDecodingParams(json=sample_json_schema,
                                        backend=backend)
    json_lp = get_local_guided_decoding_logits_processor(
        json_request, deepseek_r1_qwen_tokenizer, config,
        reasoning_backend) if is_local else \
        await get_guided_decoding_logits_processor(
            json_request, deepseek_r1_qwen_tokenizer, config, reasoning_backend)
    assert json_lp is not None
    tensor = torch.rand(32000)
    original_tensor = torch.clone(tensor)
    tensor = json_lp(token_ids, tensor)
    assert tensor.shape == original_tensor.shape
    assert torch.allclose(tensor, original_tensor)

    # Thinking is over, so the tensor should change.
    token_ids = deepseek_r1_qwen_tokenizer.encode(
        f"Give an employee profile that fits this schema: {sample_json_schema}."
        "<think>here is the thinking process</think> Then")
    json_request = GuidedDecodingParams(json=sample_json_schema,
                                        backend=backend)
    json_lp = get_local_guided_decoding_logits_processor(
        json_request, deepseek_r1_qwen_tokenizer, config,
        reasoning_backend) if is_local else \
        await get_guided_decoding_logits_processor(
            json_request, deepseek_r1_qwen_tokenizer, config, reasoning_backend)
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
    with pytest.warns(DeprecationWarning):
        guided_decoding_params = GuidedDecodingParams(
            backend=
            "xgrammar:no-fallback,disable-any-whitespace,no-additional-properties"
        )
    assert guided_decoding_params.backend == "xgrammar"
    assert guided_decoding_params.disable_fallback
    assert guided_decoding_params.disable_any_whitespace
    assert guided_decoding_params.disable_additional_properties


def test_pickle_xgrammar_tokenizer_data():
    try:
        import xgrammar as xgr
    except ImportError:
        pytest.skip("Could not import xgrammar to run test")

    from vllm.model_executor.guided_decoding.xgrammar_decoding import (
        TokenizerData)
    tokenizer_data = TokenizerData(
        metadata=
        '{"vocab_type":2,"vocab_size":151665,"add_prefix_space":false,"stop_token_ids":[151645]}',
        encoded_vocab=['!', '"', '#', '$', '%'],
    )
    pickled = pickle.dumps(tokenizer_data)

    assert pickled is not None

    depickled: TokenizerData = pickle.loads(pickled)

    assert depickled is not None
    assert json.loads(
        depickled.metadata)['vocab_type'] == xgr.VocabType.BYTE_LEVEL.value

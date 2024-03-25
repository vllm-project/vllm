import asyncio
import os
from unittest.mock import patch

import pytest
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from vllm.transformers_utils.tokenizer_group import get_tokenizer_group
from vllm.transformers_utils.tokenizer_group.ray_tokenizer_group import (
    RayTokenizerGroupPool)
from vllm.transformers_utils.tokenizer_group.tokenizer_group import (
    TokenizerGroup)

from ..conftest import get_tokenizer_pool_config


@pytest.mark.asyncio
@pytest.mark.parametrize("tokenizer_group_type", [None, "ray"])
async def test_tokenizer_group(tokenizer_group_type):
    reference_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer_group = get_tokenizer_group(
        get_tokenizer_pool_config(tokenizer_group_type),
        tokenizer_id="gpt2",
        enable_lora=False,
        max_num_seqs=1,
        max_input_length=None,
    )
    assert reference_tokenizer.encode("prompt") == tokenizer_group.encode(
        request_id="request_id", prompt="prompt", lora_request=None)
    assert reference_tokenizer.encode(
        "prompt") == await tokenizer_group.encode_async(
            request_id="request_id", prompt="prompt", lora_request=None)
    assert isinstance(tokenizer_group.get_lora_tokenizer(None),
                      PreTrainedTokenizerBase)
    assert tokenizer_group.get_lora_tokenizer(
        None) == await tokenizer_group.get_lora_tokenizer_async(None)


@pytest.mark.asyncio
@pytest.mark.parametrize("tokenizer_group_type", ["ray"])
async def test_tokenizer_group_pool(tokenizer_group_type):
    reference_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer_group_pool = get_tokenizer_group(
        get_tokenizer_pool_config(tokenizer_group_type),
        tokenizer_id="gpt2",
        enable_lora=False,
        max_num_seqs=1,
        max_input_length=None,
    )
    # Send multiple requests to the tokenizer group pool
    # (more than the pool size)
    # and check that all requests are processed correctly.
    num_requests = tokenizer_group_pool.pool_size * 5
    requests = [
        tokenizer_group_pool.encode_async(request_id=str(i),
                                          prompt=f"prompt {i}",
                                          lora_request=None)
        for i in range(num_requests)
    ]
    results = await asyncio.gather(*requests)
    expected_results = [
        reference_tokenizer.encode(f"prompt {i}") for i in range(num_requests)
    ]
    assert results == expected_results


@pytest.mark.asyncio
@pytest.mark.parametrize("tokenizer_group_type", ["ray"])
async def test_tokenizer_group_ray_pool_env_var_propagation(
        tokenizer_group_type):
    """Test that env vars from caller process are propagated to
    tokenizer Ray actors."""
    env_var = "MY_ENV_VAR"

    class EnvVarCheckerTokenizerGroup(TokenizerGroup):

        def ping(self):
            assert os.environ.get(env_var) == "1"
            return super().ping()

    class EnvVarCheckerRayTokenizerGroupPool(RayTokenizerGroupPool):
        _worker_cls = EnvVarCheckerTokenizerGroup

    tokenizer_pool_config = get_tokenizer_pool_config(tokenizer_group_type)
    tokenizer_pool = EnvVarCheckerRayTokenizerGroupPool.from_config(
        tokenizer_pool_config,
        tokenizer_id="gpt2",
        enable_lora=False,
        max_num_seqs=1,
        max_input_length=None)
    with pytest.raises(AssertionError):
        tokenizer_pool.ping()

    with patch.dict(os.environ, {env_var: "1"}):
        tokenizer_pool_config = get_tokenizer_pool_config(tokenizer_group_type)
        tokenizer_pool = EnvVarCheckerRayTokenizerGroupPool.from_config(
            tokenizer_pool_config,
            tokenizer_id="gpt2",
            enable_lora=False,
            max_num_seqs=1,
            max_input_length=None)
        tokenizer_pool.ping()

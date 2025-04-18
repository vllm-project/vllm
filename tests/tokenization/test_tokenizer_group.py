# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
import sys
from typing import List, Optional
from unittest.mock import patch

import pytest
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from vllm.transformers_utils.tokenizer_group import (TokenizerGroup,
                                                     get_tokenizer_group)
from vllm.transformers_utils.tokenizer_group.ray_tokenizer_group import (
    RayTokenizerGroupPool)

from ..conftest import get_tokenizer_pool_config


class CustomTokenizerGroup(TokenizerGroup):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._i = 0

    def encode(self, *args, **kwargs):
        self._i += 1
        return super().encode(*args, **kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize("tokenizer_group_type",
                         [None, "ray", CustomTokenizerGroup])
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
    if tokenizer_group_type is CustomTokenizerGroup:
        assert tokenizer_group._i > 0


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


@pytest.mark.asyncio
@pytest.mark.parametrize("tokenizer_group_type", ["ray"])
async def test_tokenizer_group_ray_pool_fault_tolerance(tokenizer_group_type):
    """Test that Ray tokenizer pool group can recover from failures and
    if that's not possible, mark itself as unhealthy."""

    class FailingTokenizerGroup(TokenizerGroup):

        def __init__(self,
                     *args,
                     fail_at: Optional[List[int]] = None,
                     **kwargs):
            super().__init__(*args, **kwargs)
            self.i = 0
            self.fail_at = fail_at or []

        def encode(self, *args, **kwargs):
            self.i += 1
            if self.i in self.fail_at:
                sys.exit(1)
            return super().encode(*args, **kwargs)

    class FailingRayTokenizerGroupPool(RayTokenizerGroupPool):
        _worker_cls = FailingTokenizerGroup

    # Fail at first iteration
    fail_at = [1]
    tokenizer_pool_config = get_tokenizer_pool_config(tokenizer_group_type)
    tokenizer_group_pool = FailingRayTokenizerGroupPool.from_config(
        tokenizer_pool_config,
        tokenizer_id="gpt2",
        enable_lora=False,
        max_num_seqs=1,
        max_input_length=None,
        fail_at=fail_at)
    tokenizer_actors = tokenizer_group_pool.tokenizer_actors.copy()

    # Modify fail at to not fail at all (will be re-read when actor is
    # re-initialized).
    fail_at[0] = 1000

    # We should recover successfully.
    await tokenizer_group_pool.encode_async(request_id="1",
                                            prompt="prompt",
                                            lora_request=None)
    await tokenizer_group_pool.encode_async(request_id="1",
                                            prompt="prompt",
                                            lora_request=None)

    # Check that we have a new actor
    assert len(tokenizer_group_pool.tokenizer_actors) == len(tokenizer_actors)
    assert tokenizer_group_pool.tokenizer_actors != tokenizer_actors

    # Fail at first iteration
    fail_at = [1]
    tokenizer_group_pool = FailingRayTokenizerGroupPool.from_config(
        tokenizer_pool_config,
        tokenizer_id="gpt2",
        enable_lora=False,
        max_num_seqs=1,
        max_input_length=None,
        fail_at=fail_at)

    # We should fail after re-initialization.
    with pytest.raises(RuntimeError):
        await tokenizer_group_pool.encode_async(request_id="1",
                                                prompt="prompt",
                                                lora_request=None)

    # check_health should raise the same thing
    with pytest.raises(RuntimeError):
        tokenizer_group_pool.check_health()

    # Ensure that non-ActorDiedErrors are still propagated correctly and do not
    # cause a re-initialization.
    fail_at = []
    tokenizer_group_pool = FailingRayTokenizerGroupPool.from_config(
        tokenizer_pool_config,
        tokenizer_id="gpt2",
        enable_lora=False,
        max_num_seqs=1,
        max_input_length=2,
        fail_at=fail_at)
    tokenizer_actors = tokenizer_group_pool.tokenizer_actors.copy()

    # Prompt too long error
    with pytest.raises(ValueError):
        await tokenizer_group_pool.encode_async(request_id="1",
                                                prompt="prompt" * 100,
                                                lora_request=None)
    await tokenizer_group_pool.encode_async(request_id="1",
                                            prompt="prompt",
                                            lora_request=None)
    # Actors should stay the same.
    assert tokenizer_group_pool.tokenizer_actors == tokenizer_actors

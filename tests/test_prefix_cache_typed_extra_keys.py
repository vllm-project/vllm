# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_utils import generate_block_hash_extra_keys
from vllm.v1.request import Request

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


def _make_request(request_id: str, cache_salt: str | None = None) -> Request:
    return Request(
        request_id=request_id,
        prompt_token_ids=[0, 1, 2],
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        cache_salt=cache_salt,
    )


def test_lora_name_and_cache_salt_use_distinct_prefix_cache_keys():
    lora_request = _make_request("lora")
    lora_request.lora_request = LoRARequest(
        lora_name="foo", lora_int_id=1, lora_path="/path/to/lora"
    )
    salted_request = _make_request("salted", cache_salt="foo")

    lora_extra_keys, _ = generate_block_hash_extra_keys(lora_request, 0, 3, 0)
    salted_extra_keys, _ = generate_block_hash_extra_keys(salted_request, 0, 3, 0)

    assert lora_extra_keys == (("lora", "foo"),)
    assert salted_extra_keys == (("cache_salt", "foo"),)
    assert lora_extra_keys != salted_extra_keys

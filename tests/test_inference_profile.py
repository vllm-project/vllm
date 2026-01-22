# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.engine import InferenceProfile
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.request import Request


def test_inference_profile_propagation():
    profile = InferenceProfile(
        mode="interactive",
        max_latency_ms=500,
        priority=10,
    )

    sampling_params = SamplingParams(max_tokens=1)

    core_req = EngineCoreRequest(
        request_id="test-id",
        prompt_token_ids=[],
        mm_features=None,
        sampling_params=sampling_params,
        pooling_params=None,
        eos_token_id=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        inference_profile=profile,
    )

    req = Request.from_engine_core_request(core_req, block_hasher=None)

    assert req.inference_profile is profile

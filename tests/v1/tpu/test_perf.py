# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A basic performance regression test for TPUs

Run `pytest tests/v1/tpu/test_perf.py`.
"""

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pytest

from vllm.platforms import current_platform
from vllm.sampling_params import SamplingParams
from vllm.tokenizers import get_tokenizer

if TYPE_CHECKING:
    from tests.conftest import VllmRunner
else:
    VllmRunner = object


@dataclass
class TestParams:
    model: str
    num_prompts: int
    prefix_len: int
    decode_len: int
    expected_avg_time: float
    err_tol: float


TEST_PARAMS = [
    # TODO: Cannot run a series of tests because:
    #   RuntimeError: Bad StatusOr access: UNKNOWN: TPU initialization failed:
    #   open(/dev/vfio/0): Device or resource busy: Device or resource busy;
    #   Couldn't open iommu group /dev/vfio/0
    # => Investigate
    # TestParams(
    #     model="Qwen/Qwen2.5-1.5B-Instruct",
    #     num_prompts=1,
    #     prefix_len=10,
    #     decode_len=5,
    #     expected_avg_time=0.03,
    #     err_tol=0.01,
    # ),
    # TestParams(
    #     model="Qwen/Qwen2.5-1.5B-Instruct",
    #     num_prompts=10,
    #     prefix_len=100,
    #     decode_len=50,
    #     expected_avg_time=0.234,
    #     err_tol=0.020,
    # ),
    TestParams(
        model="Qwen/Qwen2.5-1.5B-Instruct",
        num_prompts=64,
        prefix_len=500,
        decode_len=50,
        # commit id: ccb246776d93ef105904a8ec015b3587240a1183
        # tpu: v5lite (old vllm CI/CD)
        # expected_avg_time=1.4,
        # err_tol=0.30,
        # (This is the active CI/CD instance)
        # commit id: ccb246776d93ef105904a8ec015b3587240a1183
        # tpu: v6e (current vllm CI/CD)
        expected_avg_time=1.7,  # measured with VLLM_XLA_CACHE_PATH=
        err_tol=0.20,
    ),
]

NUM_WARMUPS = 5
NUM_RUNS = 10

MAX_MODEL_LEN = 1024
MAX_NUM_SEQS = 32
GPU_UTIL = 0.9


@pytest.mark.skipif(
    not current_platform.is_tpu(),
    reason="This is a basic performance test for TPU only",
)
@pytest.mark.parametrize("params", TEST_PARAMS)
def test_perf(
    vllm_runner: type[VllmRunner],
    params: TestParams,
) -> None:
    tokenizer = get_tokenizer(
        params.model, tokenizer_mode="auto", trust_remote_code=True
    )

    prompts = []
    for i in range(params.num_prompts):
        prefix_token_ids = np.random.randint(
            0, tokenizer.vocab_size, size=params.prefix_len
        ).tolist()
        prompt = tokenizer.decode(prefix_token_ids)
        prompts.append(prompt)

    print(
        "-- Running: num_prompts = {} prefix_len = {} decode_len = {}".format(
            len(prompts), params.prefix_len, params.decode_len
        )
    )

    sampling_params = SamplingParams(
        max_tokens=params.decode_len, temperature=1.0, min_p=0.0
    )

    with vllm_runner(
        params.model,
        max_num_batched_tokens=MAX_MODEL_LEN,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=MAX_NUM_SEQS,
        gpu_memory_utilization=GPU_UTIL,
        enforce_eager=False,
        tensor_parallel_size=1,
    ) as vllm_model:
        print("  -- Warmup / Compile")
        for i in range(NUM_WARMUPS):
            _ = vllm_model.generate(prompts, sampling_params)

        print("  -- Benchmarking... ")
        times = []
        for i in range(NUM_RUNS):
            start_time = time.time()
            _ = vllm_model.generate(prompts, sampling_params)
            times.append(time.time() - start_time)

        avg_time = sum(times) / len(times)

        print("  -- avg_time = {}".format(avg_time))
        print(
            "  -- expected_avg_time = {} with err_tol = {}".format(
                params.expected_avg_time, params.err_tol
            )
        )
        diff = avg_time - params.expected_avg_time
        ok = diff < params.err_tol
        if diff < -params.err_tol:
            print(
                "  !! WARNING !! Performance has improved by {}, "
                "it may be necessary to fine-tune the "
                "expected_avg_time = {}".format(-diff, params.expected_avg_time)
            )

        assert ok, " !! ERROR !! Regression detected"

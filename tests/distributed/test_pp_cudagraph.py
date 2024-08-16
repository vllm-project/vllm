import os

import pytest

from ..utils import compare_two_settings, fork_new_process_for_each_test


@pytest.mark.parametrize("PP_SIZE, MODEL_NAME", [
    (2, "JackFram/llama-160m"),
])
@pytest.mark.parametrize("ATTN_BACKEND", [
    "FLASH_ATTN",
    "FLASHINFER",
])
@fork_new_process_for_each_test
def test_pp_cudagraph(PP_SIZE, MODEL_NAME, ATTN_BACKEND):
    cudagraph_args = [
        # use half precision for speed and memory savings in CI environment
        "--dtype",
        "float16",
        "--pipeline-parallel-size",
        str(PP_SIZE),
        "--distributed-executor-backend",
        "mp",
    ]
    os.environ["VLLM_ATTENTION_BACKEND"] = ATTN_BACKEND

    eager_args = cudagraph_args + ["--enforce-eager"]

    compare_two_settings(MODEL_NAME, eager_args, cudagraph_args)

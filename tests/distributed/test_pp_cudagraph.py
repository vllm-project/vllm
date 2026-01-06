# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
from typing_extensions import LiteralString

from ..utils import compare_two_settings, create_new_process_for_each_test


@pytest.mark.parametrize(
    "PP_SIZE, MODEL_NAME",
    [
        (2, "JackFram/llama-160m"),
    ],
)
@pytest.mark.parametrize(
    "ATTN_BACKEND",
    [
        "FLASH_ATTN",
    ],
)
@create_new_process_for_each_test()
def test_pp_cudagraph(
    PP_SIZE: int,
    MODEL_NAME: str,
    ATTN_BACKEND: LiteralString,
):
    cudagraph_args = [
        # use half precision for speed and memory savings in CI environment
        "--dtype",
        "float16",
        "--pipeline-parallel-size",
        str(PP_SIZE),
        "--distributed-executor-backend",
        "mp",
        f"--attention-backend={ATTN_BACKEND}",
    ]

    eager_args = cudagraph_args + ["--enforce-eager"]

    compare_two_settings(MODEL_NAME, eager_args, cudagraph_args)
